"""
================================================================================
PARAMETER GOLF SUBMISSION — Jerry's stacked baseline
================================================================================

This is a modified version of the official baseline `train_gpt.py`. The high-level
goal: free up enough artifact bytes (via int6 quant + zstd-22) to run a deeper
model (11 layers, mlp_mult=3) at sequence length 2048, then squeeze additional
BPB out of the model with three near-free architectural / training tricks
(Partial RoPE, LN Scale, EMA), plus a free post-training quantization optimization
(GPTQ-lite per-row clip search) and a free eval-time win (sliding window eval).

Read this top section before touching anything. Each subsection explains what
changed, why, and how to disable it if it breaks.

--------------------------------------------------------------------------------
WHAT THIS SCRIPT EXPECTS YOU TO RUN
--------------------------------------------------------------------------------

For a single 8xH100 final submission run:

    SEED=1337 \
    NUM_LAYERS=11 MODEL_DIM=512 MLP_MULT=3 NUM_HEADS=8 NUM_KV_HEADS=4 \
    TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
    ITERATIONS=15000 WARMDOWN_ITERS=3500 \
    MAX_WALLCLOCK_SECONDS=580 \
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 \
    EMA_ENABLED=1 EMA_DECAY=0.997 \
    PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
    QUANT_BITS=6 GPTQ_LITE=1 USE_ZSTD=1 ZSTD_LEVEL=22 \
    SLIDING_EVAL=1 SLIDING_STRIDE=64 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py

For a 1xH100 smoke test (cheaper, faster, lower BPB but proves it runs):

    SEED=1337 \
    NUM_LAYERS=11 MLP_MULT=3 \
    TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=98304 \
    MAX_WALLCLOCK_SECONDS=580 \
    EMA_ENABLED=1 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
    QUANT_BITS=6 GPTQ_LITE=1 USE_ZSTD=1 \
    SLIDING_EVAL=1 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py

To go back to a behavior-identical baseline (kill switch for everything new):

    EMA_ENABLED=0 PARTIAL_ROPE_DIMS=0 LN_SCALE=0 \
    QUANT_BITS=8 GPTQ_LITE=0 USE_ZSTD=0 \
    SLIDING_EVAL=0 \
    torchrun ...

--------------------------------------------------------------------------------
THE CHANGES — IN ORDER OF CONFIDENCE
--------------------------------------------------------------------------------

[1] HIGH CONFIDENCE: Env-var hyperparameter changes (no code logic changed)

    NUM_LAYERS 9 -> 11           : more depth, paid for by int6 quant freeing space
    MLP_MULT   2 -> 3            : wider MLP, also paid for by int6
    TRAIN_SEQ_LEN 1024 -> 2048   : proven leaderboard win, longer context
    WARMDOWN_ITERS 1200 -> 3500  : proven leaderboard win, longer LR decay tail
    MUON_MOMENTUM 0.95 -> 0.99   : matches SOTA configs
    MUON_MOMENTUM_WARMUP_START 0.85 -> 0.92
    MUON_MOMENTUM_WARMUP_STEPS 500 -> 1500
    MATRIX_LR  0.04 -> 0.025     : SOTA configs converged on lower LR for 11L
    SCALAR_LR  0.04 -> 0.025
    TIED_EMBED_LR 0.05 -> 0.035

    These are zero-risk changes — they don't touch the code, just the values.
    If anything misbehaves, the env vars are the first place to roll back.

[2] HIGH CONFIDENCE: EMA (Exponential Moving Average) of weights.

    Toggled by EMA_ENABLED=1. Decay=0.997 (env: EMA_DECAY).

    A shadow copy of the model weights is updated every step:
        ema = decay * ema + (1 - decay) * current_weights
    At quantization time, we swap the live weights for the EMA weights.
    Costs ~zero compute, costs one extra weight tensor in CPU/GPU memory.
    Worth ~0.0006 BPB in the SOTA submission.

    Implementation: `EMAShadow` class. Lives outside the DDP/compile graph —
    we operate on `base_model`'s parameters directly, never the compiled or
    DDP-wrapped one. The torch.compile / DDP gotcha for EMA is that if you
    update via the compiled graph, the EMA tensor either gets compiled into
    constants or doesn't sync across ranks. We sidestep both.

    KILL SWITCH: EMA_ENABLED=0 disables it entirely.

[3] HIGH CONFIDENCE: Partial RoPE (16 of 64 head dims).

    Toggled by PARTIAL_ROPE_DIMS=16. Set to 64 (full) or 0 (no rope) to disable.

    Standard RoPE rotates ALL head dimensions by frequency-encoded angles. This
    forces every attention head to be position-aware. Partial RoPE rotates only
    the first PARTIAL_ROPE_DIMS dims (16 out of 64), leaving the other 48 dims
    untouched. Heads can now learn position-invariant attention patterns when
    that's what's useful (e.g. content-based retrieval) while still having
    positional info available where needed.

    Zero new params. Worth ~0.001-0.002 BPB.

    Implementation: `apply_partial_rotary_emb` splits the head dim into a
    rotated portion (first N dims) and a passthrough portion (remaining dims),
    then concatenates them back. The Rotary cache is sized to N, not full
    head_dim, so the cos/sin tables are smaller too.

    KILL SWITCH: PARTIAL_ROPE_DIMS=64 reverts to original full RoPE behavior.

[4] HIGH CONFIDENCE: LN Scale (depth-damped residual contributions).

    Toggled by LN_SCALE=1. Each Block scales its attn output and mlp output
    by 1/sqrt(layer_idx + 1). Deep layers contribute less to the residual
    stream, especially at init, which stabilizes training in deeper networks.
    Zero new params.

    *** TORCH.COMPILE GOTCHA ***
    The PR #315 winning submission had this idea coded as a Python class
    attribute that torch.compile constant-folded into a no-op. Nobody noticed
    for days. Lesson: do not store layer-dependent scalars as Python ints
    you read at forward time. We bake the scale as a registered float buffer
    on each Block at construction time, so it's a tensor that survives compile.

    KILL SWITCH: LN_SCALE=0 disables it (scale becomes 1.0).

[5] MEDIUM CONFIDENCE — HIGHEST DEBUG RISK: Int6 quantization.

    Toggled by QUANT_BITS=6. Set to 8 for the original int8 path.

    The whole point of this submission. Int6 cuts the per-weight cost from 8
    bits to 6 bits, which is a 25% reduction in weight bytes. That's the
    headroom we spend on going from 9 layers to 11, and from mlp_mult=2 to 3.

    Bit packing: we pack 4 int6 values (range [-32, 31] after centering) into
    3 bytes (24 bits). For a row of W weights:
      - quantize each weight to int6: round(clip(w / scale)) -> int in [-32, 31]
      - shift to unsigned by adding 32: now in [0, 63]
      - pack groups of 4 unsigned-int6 values into 3 bytes
      - if W is not divisible by 4, pad the row with zeros at the end and
        store the padding count in metadata so dequant trims correctly

    What's still int8 vs int6:
      - Embeddings: int8 (sensitive — quantization errors propagate through
        every token lookup; the leaderboard universally keeps these int8)
      - Matrix weights (attn Q/K/V/O, MLP fc/proj): int6
      - Scales, norms, control tensors (resid_mix, q_gain, scale params,
        skip_weights): kept fp32 or fp16 as in the baseline.

    All of this only changes the EXPORTED artifact format. Training is still
    bf16/fp32. Decompression at eval time reconstructs the float weights from
    the packed int6 + float scale.

    KILL SWITCH: QUANT_BITS=8 reverts to per-row int8 (the baseline path).
    If the int6 roundtrip BPB looks insane (>2.0 or NaN), this is the first
    place to look.

[6] LOW CONFIDENCE / NICE-TO-HAVE: GPTQ-lite per-row clip search.

    Toggled by GPTQ_LITE=1. For each row of each int6-quantized matrix, try
    5 clip percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0). For each
    percentile, quantize-then-dequantize the row, measure MSE vs the original
    row, and keep the percentile that minimizes MSE. The result is a
    per-row scale that may clip aggressively when that helps and not when it
    doesn't. Worth ~0.0006 BPB. Costs only a few seconds at quant time.

    KILL SWITCH: GPTQ_LITE=0 reverts to the simpler row-max scale.

[7] LOW RISK: zstd-22 compression instead of zlib-9.

    Toggled by USE_ZSTD=1, level via ZSTD_LEVEL=22. zstd at level 22 hits
    a meaningfully better compression ratio than zlib-9 on the int6 byte
    stream. Requires `pip install zstandard` (which is in the Runpod image
    listed in requirements.txt for these challenges).

    Auto-falls-back to zlib if `zstandard` isn't importable, so the script
    runs anywhere.

    KILL SWITCH: USE_ZSTD=0 reverts to zlib level 9.

[8] LOW RISK / EVAL ONLY: Sliding window eval.

    Toggled by SLIDING_EVAL=1, stride via SLIDING_STRIDE (default 64).

    The default eval chops val tokens into disjoint chunks of length seq_len.
    Tokens at the start of each chunk get little-to-no preceding context.
    Sliding window eval moves a window of size seq_len across the val stream
    with a stride < seq_len. Each token is scored only when it has had at
    least (seq_len - stride) tokens of context. Worth ~0.02 BPB just from
    measurement methodology — a free score boost without changing the model.

    Costs more eval time (roughly seq_len/stride more compute), but eval is
    not subject to the 10-minute training cap. Final post-quant eval uses
    sliding when SLIDING_EVAL=1; intermediate eval during training uses the
    fast non-sliding path so we don't waste training time.

    KILL SWITCH: SLIDING_EVAL=0 reverts to the original disjoint-chunk eval.

--------------------------------------------------------------------------------
WHAT TO LOOK AT IF THINGS GO WRONG
--------------------------------------------------------------------------------

Symptom: train_loss is NaN or exploding from step 1.
  -> Probably MATRIX_LR/SCALAR_LR too high for current model size, or
     LN_SCALE is wrong direction. Try LN_SCALE=0 first.

Symptom: int6 roundtrip BPB is way worse than pre-quant BPB (gap > 0.05).
  -> The int6 packer / unpacker has a bug. Set QUANT_BITS=8 and verify.
     Common cause: row length not divisible by 4 and padding metadata not
     restored correctly.

Symptom: artifact size > 16,000,000 bytes.
  -> Either (a) too many layers / too wide MLP for the chosen QUANT_BITS,
     or (b) zstd not actually being used. Look for the
     "Total submission size" line in logs.

Symptom: EMA seems to do nothing (BPB identical with EMA on/off).
  -> The shadow probably isn't being copied into the model before quant.
     Check the `swap_in_ema_weights()` call right before quantization.

Symptom: LN Scale silently disabled by torch.compile.
  -> Print `block.ln_scale` before and after a forward pass. If both show
     1.0 you're hitting the compile gotcha. Should be 1/sqrt(i+1) per layer.

================================================================================
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Optional zstd compression — falls back to zlib if not available.
try:
    import zstandard as zstd  # type: ignore
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    # Data paths.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length. Defaults are tuned for 8xH100 SOTA-stack runs; override
    # TRAIN_BATCH_TOKENS down to 98304 or so for 1xH100 smoke tests.
    iterations = int(os.environ.get("ITERATIONS", 15000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 580.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape — bigger than baseline (paid for by int6 quant).
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # New: Partial RoPE. 0 disables RoPE entirely (useful only for ablation).
    # 64 means full RoPE (the original baseline behavior).
    # 16 means rotate first 16 dims of each 64-dim head (default for SOTA stack).
    partial_rope_dims = int(os.environ.get("PARTIAL_ROPE_DIMS", 16))

    # New: LN Scale. 1 enables 1/sqrt(layer+1) scaling on attn/mlp outputs.
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))

    # Optimizer hyperparameters — tuned for 11L SOTA configs.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # New: EMA.
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    # Start EMA after this many steps to avoid corrupting it with random init weights.
    ema_warmup_steps = int(os.environ.get("EMA_WARMUP_STEPS", 100))

    # New: Quantization controls.
    quant_bits = int(os.environ.get("QUANT_BITS", 6))  # 8 or 6
    gptq_lite = bool(int(os.environ.get("GPTQ_LITE", "1")))
    use_zstd = bool(int(os.environ.get("USE_ZSTD", "1")))
    zstd_level = int(os.environ.get("ZSTD_LEVEL", 22))

    # New: Sliding window eval.
    sliding_eval = bool(int(os.environ.get("SLIDING_EVAL", "1")))
    sliding_stride = int(os.environ.get("SLIDING_STRIDE", 64))


# -----------------------------
# MUON OPTIMIZER (unchanged from baseline)
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP (mostly unchanged; sliding-window logic added)
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Fast disjoint-chunk eval used during training. Same logic as baseline."""
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
) -> tuple[float, float]:
    """
    Sliding window eval. Each scoring window has length seq_len, and the
    "scored" portion is the last `stride` tokens — i.e. tokens with the most
    preceding context. Windows advance by `stride` tokens. The first window
    is special: we score all of it because there's no earlier context to use.

    Total scored tokens = (val_tokens.numel() - 1) (every token gets scored once).
    Total compute ~ seq_len/stride * baseline.

    Tokens are partitioned across ranks by window index for distributed eval.
    """
    seq_len = args.train_seq_len
    n_total = val_tokens.numel() - 1  # last token has no "next" target
    if n_total <= seq_len:
        # Fall back to non-sliding for very short val sets.
        return eval_val(args, model, rank, world_size, device, 1, val_tokens,
                        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)

    # Window starts: 0, stride, 2*stride, ...
    # First window covers tokens [0, seq_len), scores all of them.
    # Each subsequent window covers [start, start+seq_len), scores last `stride`.
    # The very last window may extend beyond n_total; we clamp.
    starts = list(range(0, n_total - seq_len + 1, stride))
    # Make sure we score the tail.
    if not starts or starts[-1] + seq_len < n_total:
        starts.append(max(n_total - seq_len, 0))

    # Distribute windows across ranks.
    my_starts = starts[rank::world_size]

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for i, start in enumerate(my_starts):
            end = min(start + seq_len, n_total)
            local = val_tokens[start : end + 1].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(1, -1)
            y = local[1:].reshape(1, -1)
            T = x.size(1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                # Model returns mean cross entropy over all positions of (x, y).
                # We need per-position losses to score only the tail. Recompute:
                per_pos_loss = _model_per_position_loss(model, x, y)  # shape (T,)

            # Determine which positions to score.
            if i == 0 and start == 0:
                # First window: score everything.
                score_lo, score_hi = 0, T
            else:
                # Score only the last `stride` positions (newest tokens with most context).
                score_lo = max(0, T - stride)
                score_hi = T

            scored_loss = per_pos_loss[score_lo:score_hi].to(torch.float64).sum()
            scored_tokens = score_hi - score_lo
            val_loss_sum += scored_loss
            val_token_count += scored_tokens

            # Bytes for those scored positions:
            tgt_ids = y[0, score_lo:score_hi]
            prev_ids = x[0, score_lo:score_hi]
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def _model_per_position_loss(model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """
    Sliding-window eval needs per-position losses, but the GPT.forward returns
    a mean. We unwrap one layer to get logits and recompute per-position CE.

    The model is wrapped (DDP / compile / both). `_underlying_gpt` walks
    through to find the `GPT` instance. Then we call its forward but bypass
    the mean reduction.
    """
    gpt = _underlying_gpt(model)
    return gpt.per_position_loss(x, y)


def _underlying_gpt(model: nn.Module) -> "GPT":
    m = model
    # DDP -> compiled -> GPT, or compiled -> GPT, or just GPT.
    while hasattr(m, "module"):
        m = m.module  # type: ignore[attr-defined]
    while hasattr(m, "_orig_mod"):
        m = m._orig_mod  # type: ignore[attr-defined]
    return m  # type: ignore[return-value]


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# This section is the heart of the artifact-size win. The baseline used per-row
# int8. We add:
#   - QUANT_BITS=6 path with bit packing 4 int6 values into 3 bytes.
#   - GPTQ-lite per-row clip percentile search.
#   - zstd-22 instead of zlib-9 for the final compression.
#   - Embeddings still kept at int8 (sensitive).

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,ln_scale_buf",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT_PER_ROW_SCALE_DTYPE = torch.float16

# Embeddings always stay int8 even when QUANT_BITS=6; matched by name.
EMBEDDING_NAME_PATTERNS = ("tok_emb",)


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def _pack_int6_row(row_int: np.ndarray) -> tuple[bytes, int]:
    """
    Pack a 1D array of int6 values (already centered to range [0, 63] as uint8)
    into a tightly-packed byte string. Returns (packed_bytes, padding_count).

    Packing: 4 values * 6 bits = 24 bits = 3 bytes per group.
      byte0 = v0 | (v1 << 6)            <- low 6 bits of v0, low 2 of v1 in top 2
            actually:  v0 in bits 0..5, v1 low 2 bits in bits 6..7
      byte1 = (v1 >> 2) | (v2 << 4)
            v1 high 4 bits in bits 0..3, v2 low 4 bits in bits 4..7
      byte2 = (v2 >> 4) | (v3 << 2)
            v2 high 2 bits in bits 0..1, v3 in bits 2..7
    """
    n = row_int.size
    pad = (4 - (n % 4)) % 4
    if pad:
        row_int = np.concatenate([row_int, np.zeros(pad, dtype=np.uint8)])
    quads = row_int.reshape(-1, 4).astype(np.uint16)  # uint16 to avoid overflow during shift+or
    v0, v1, v2, v3 = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    b0 = (v0 | (v1 << 6)) & 0xFF
    b1 = ((v1 >> 2) | (v2 << 4)) & 0xFF
    b2 = ((v2 >> 4) | (v3 << 2)) & 0xFF
    out = np.stack([b0, b1, b2], axis=1).astype(np.uint8).tobytes()
    return out, pad


def _unpack_int6_row(packed: bytes, n: int, pad: int) -> np.ndarray:
    """Inverse of _pack_int6_row. Returns an array of n unsigned-int6 values in [0, 63]."""
    arr = np.frombuffer(packed, dtype=np.uint8).reshape(-1, 3).astype(np.uint16)
    b0, b1, b2 = arr[:, 0], arr[:, 1], arr[:, 2]
    v0 = b0 & 0x3F
    v1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
    v2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
    v3 = (b2 >> 2) & 0x3F
    out = np.stack([v0, v1, v2, v3], axis=1).reshape(-1).astype(np.uint8)
    if pad:
        out = out[:-pad] if pad <= out.size else out[:0]
    assert out.size == n, f"unpack mismatch: got {out.size} expected {n}"
    return out


def _gptq_lite_scale_int(row_f32: Tensor, q_max: int) -> Tensor:
    """
    Per-row: try 5 clip percentiles and pick the one with min reconstruction MSE.
    q_max is the max signed integer for the chosen bit width (127 for int8, 31 for int6).
    Returns scalar scale.
    """
    abs_row = row_f32.abs()
    if abs_row.numel() == 0:
        return torch.tensor(1.0 / q_max, dtype=torch.float32)
    candidates = [
        torch.quantile(abs_row, q).item() for q in (0.999, 0.9995, 0.9999, 0.99999, 1.0)
    ]
    candidates = [c for c in candidates if c > 0]
    if not candidates:
        return torch.tensor(1.0 / q_max, dtype=torch.float32)

    best_scale = None
    best_mse = float("inf")
    for clip in candidates:
        scale = clip / q_max
        if scale <= 0:
            continue
        clipped = row_f32.clamp(-clip, clip)
        q = torch.round(clipped / scale).clamp(-q_max, q_max)
        recon = q * scale
        mse = (recon - row_f32).square().mean().item()
        if mse < best_mse:
            best_mse = mse
            best_scale = scale
    return torch.tensor(best_scale if best_scale is not None else candidates[-1] / q_max,
                        dtype=torch.float32)


def quantize_float_tensor(t: Tensor, bits: int, gptq_lite: bool) -> tuple[Tensor, Tensor, dict]:
    """
    Returns (quantized_payload, scales, meta). For int8, payload is int8 tensor.
    For int6, payload is a uint8 tensor of packed bytes; meta carries the
    original row length and pad count per row.
    """
    t32 = t.float()
    q_max = (1 << (bits - 1)) - 1  # 127 for int8, 31 for int6
    meta: dict = {"bits": bits}

    if t32.ndim == 2:
        rows, cols = t32.shape
        # Per-row scale.
        if gptq_lite:
            scale = torch.stack([_gptq_lite_scale_int(t32[r], q_max) for r in range(rows)])
        else:
            row_max = t32.abs().amax(dim=1).clamp_min(1e-12)
            scale = (row_max / q_max).clamp_min(1.0 / q_max)
        # Quantize.
        q_signed = torch.clamp(torch.round(t32 / scale[:, None]), -q_max, q_max).to(torch.int32)

        if bits == 8:
            payload = q_signed.to(torch.int8).contiguous()
            meta["pack"] = "int8"
            return payload, scale.to(dtype=INT_PER_ROW_SCALE_DTYPE).contiguous(), meta

        # Int6: shift to unsigned [0, 63], pack 4-into-3.
        q_unsigned = (q_signed + (q_max + 1)).to(torch.uint8).cpu().numpy()  # range [0, 63]
        # q_max+1 = 32, so signed [-31, 31] becomes unsigned [1, 63]; but we also allow -32?
        # signed range is [-q_max, q_max] = [-31, 31] -> unsigned [1, 63]. We waste 0.
        # That's fine; better to be safe than introduce off-by-one.
        packed_rows: list[bytes] = []
        pads: list[int] = []
        for r in range(rows):
            packed, pad = _pack_int6_row(q_unsigned[r])
            packed_rows.append(packed)
            pads.append(pad)
        payload = torch.tensor(np.frombuffer(b"".join(packed_rows), dtype=np.uint8),
                                dtype=torch.uint8).contiguous()
        meta["pack"] = "int6_packed"
        meta["rows"] = rows
        meta["cols"] = cols
        meta["row_pad"] = pads  # one pad value per row (0..3)
        meta["row_packed_bytes"] = [len(b) for b in packed_rows]
        meta["q_offset"] = q_max + 1
        return payload, scale.to(dtype=INT_PER_ROW_SCALE_DTYPE).contiguous(), meta

    # 1D / scalar: per-tensor scale, always int8 (these are tiny anyway).
    flat = t32.flatten()
    if flat.numel() == 0:
        return t32.to(torch.int8), torch.tensor(1.0, dtype=torch.float32), {"bits": 8, "pack": "int8_per_tensor"}
    clip_abs = float(torch.quantile(flat.abs(), 0.99999).item())
    if clip_abs <= 0:
        clip_abs = float(flat.abs().max().item()) or 1.0
    scale = torch.tensor(clip_abs / 127.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale, {"bits": 8, "pack": "int8_per_tensor"}


def quantize_state_dict(state_dict: dict[str, Tensor], args: Hyperparameters):
    """Quantize the full state dict according to args.quant_bits and args.gptq_lite.

    - 2D matrix weights: int6 packed (or int8 per-row if QUANT_BITS=8)
    - Embeddings (tok_emb): always int8 per-row (sensitive)
    - 1D / scalar floats: int8 per-tensor, or kept as fp16 if small enough
    - Non-floats: passthrough
    """
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    metas: dict[str, dict] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1

        # Embeddings always int8. Other 2D matrices follow QUANT_BITS.
        is_embedding = any(p in name for p in EMBEDDING_NAME_PATTERNS)
        bits = 8 if is_embedding else args.quant_bits

        q, s, meta = quantize_float_tensor(t, bits=bits, gptq_lite=args.gptq_lite)
        quantized[name] = q
        scales[name] = s
        metas[name] = meta
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict = {
        "__quant_format__": "mixed_int_v2",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "metas": metas,
        "passthrough": passthrough,
        "passthrough_orig_dtypes": passthrough_orig_dtypes,
    }
    return obj, stats


def dequantize_state_dict(obj: dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    metas = obj.get("metas", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, payload in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = metas.get(name, {"pack": "int8"})
        pack = meta.get("pack", "int8")

        if pack == "int8":
            # Per-row int8 2D matrix.
            scale = s.to(dtype=torch.float32)
            recon = payload.float() * scale.view(payload.shape[0], *([1] * (payload.ndim - 1)))
            out[name] = recon.to(dtype=dtype).contiguous()
        elif pack == "int8_per_tensor":
            scale = float(s.item())
            out[name] = (payload.float() * scale).to(dtype=dtype).contiguous()
        elif pack == "int6_packed":
            rows = meta["rows"]
            cols = meta["cols"]
            row_pads = meta["row_pad"]
            row_lens = meta["row_packed_bytes"]
            q_offset = meta["q_offset"]
            scale = s.to(dtype=torch.float32)

            payload_np = payload.numpy().tobytes()
            recon_rows = np.zeros((rows, cols), dtype=np.float32)
            cursor = 0
            for r in range(rows):
                blen = row_lens[r]
                packed_bytes = payload_np[cursor : cursor + blen]
                cursor += blen
                unsigned = _unpack_int6_row(packed_bytes, cols, row_pads[r])
                signed = unsigned.astype(np.int32) - q_offset
                recon_rows[r] = signed.astype(np.float32) * float(scale[r].item())
            out[name] = torch.from_numpy(recon_rows).to(dtype=dtype).contiguous()
        else:
            raise ValueError(f"Unknown pack format for {name}: {pack}")

    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def compress_blob(raw: bytes, args: Hyperparameters) -> tuple[bytes, str]:
    """zstd if available + requested, otherwise zlib. Returns (blob, codec_name)."""
    if args.use_zstd and _HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=args.zstd_level)
        return cctx.compress(raw), f"zstd-{args.zstd_level}"
    return zlib.compress(raw, level=9), "zlib-9"


def decompress_blob(blob: bytes, codec: str) -> bytes:
    if codec.startswith("zstd"):
        if not _HAS_ZSTD:
            raise RuntimeError("zstd-compressed blob but zstandard not installed")
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(blob)
    return zlib.decompress(blob)


# -----------------------------
# DATA LOADING (unchanged)
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# TRANSFORMER MODULES — Partial RoPE + LN Scale changes here
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    """
    Caches cos/sin tables. CHANGED: dim here refers to the partial-rope dim
    (number of head dims to actually rotate), not the full head dim. Caller
    is responsible for choosing the right dim.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim <= 0:
            self.register_buffer("inv_freq", torch.zeros(0), persistent=False)
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_partial_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    """
    Apply rotary embedding to only the first `rope_dims` dimensions of each
    head. The remaining (head_dim - rope_dims) dimensions are passed through
    unchanged.

    x: (B, H, T, head_dim)
    cos, sin: (1, 1, T, rope_dims // 2)  — note the half-dim shape because
                                            the rotated portion is split into
                                            two halves of size rope_dims // 2.

    The original baseline split the WHOLE head_dim in half. We split only the
    rope_dims slice in half: first rope_dims/2 dims get rotated against the
    next rope_dims/2 dims using (cos, sin), and the remaining head_dim -
    rope_dims dims are concatenated unchanged.
    """
    if rope_dims <= 0:
        return x
    if rope_dims >= x.size(-1):
        # Original behavior: rotate everything.
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

    half = rope_dims // 2
    x_rot = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((rotated, x_pass), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        partial_rope_dims: int,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        # Clamp partial_rope_dims to a valid value.
        if partial_rope_dims < 0:
            partial_rope_dims = 0
        if partial_rope_dims > self.head_dim:
            partial_rope_dims = self.head_dim
        if partial_rope_dims % 2 != 0:
            # Must be even for the rotated-pair structure.
            partial_rope_dims = partial_rope_dims - (partial_rope_dims % 2)
        self.partial_rope_dims = partial_rope_dims

        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.partial_rope_dims, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        if self.partial_rope_dims > 0:
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q = apply_partial_rotary_emb(q, cos, sin, self.partial_rope_dims)
            k = apply_partial_rotary_emb(k, cos, sin, self.partial_rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    """
    Block with two new features over baseline:
      1. Partial RoPE (passed through to CausalSelfAttention).
      2. LN Scale: each block multiplies its attn and mlp contributions by
         1/sqrt(layer_idx + 1). Stored as a registered float buffer named
         `ln_scale_buf` so it survives torch.compile.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        partial_rope_dims: int,
        layer_idx: int,
        ln_scale: bool,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, partial_rope_dims)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

        # LN Scale: bake as buffer (not Python attr) to defeat torch.compile constant-folding.
        ln_scale_value = (1.0 / math.sqrt(layer_idx + 1)) if ln_scale else 1.0
        self.register_buffer(
            "ln_scale_buf",
            torch.tensor(ln_scale_value, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        ln_s = self.ln_scale_buf.to(dtype=x.dtype)
        attn_out = self.attn(self.attn_norm(x)) * ln_s
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_out = self.mlp(self.mlp_norm(x)) * ln_s
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        partial_rope_dims: int,
        ln_scale: bool,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    partial_rope_dims,
                    layer_idx=i,
                    ln_scale=ln_scale,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _forward_to_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self._forward_to_logits(input_ids)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        return F.cross_entropy(flat_logits.float(), flat_targets, reduction="mean")

    @torch.no_grad()
    def per_position_loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """
        Returns shape (T,) of per-position cross-entropy losses for batch size 1.
        Used by sliding-window eval to score only the tail tokens of each window.
        """
        assert input_ids.size(0) == 1, "per_position_loss is for batch size 1 (sliding window eval)"
        logits = self._forward_to_logits(input_ids)
        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        flat_targets = target_ids.reshape(-1)
        return F.cross_entropy(flat_logits, flat_targets, reduction="none")


# -----------------------------
# EMA (Exponential Moving Average) — new
# -----------------------------

class EMAShadow:
    """
    Maintains a CPU-side fp32 shadow of model weights, updated as
        ema = decay * ema + (1 - decay) * current
    Updated via .update(model) once per training step (not per micro-step).

    Critical: we operate on the BASE model's parameters (not the DDP/compile
    wrapper). For DDP this is fine because all ranks see synchronized weights
    on the base model after each optimizer step. We update only on rank 0 and
    broadcast the shadow at quant time. Actually simpler: every rank maintains
    its own shadow (they're identical across ranks), and we just use rank 0's
    at quant time. We do this to avoid cross-rank sync overhead per step.

    Storage: GPU fp32, same device as the model. Memory cost is one extra copy
    of all parameters (for ~17M params, that's ~70MB — negligible).
    """
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        for name, p in model.named_parameters():
            self.shadow[name] = p.detach().clone().float()
        self.steps = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = self.decay
        for name, p in model.named_parameters():
            if name in self.shadow:
                # ema = d * ema + (1-d) * p
                self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def state_dict_for_swap(self, model: nn.Module) -> dict[str, Tensor]:
        """Return a state dict with EMA values in place of model params, but with
        non-parameter buffers etc. taken from the model. This is what we feed to
        quantize_state_dict to ensure buffers like ln_scale_buf are preserved."""
        out = {}
        sd = model.state_dict()
        for k, v in sd.items():
            if k in self.shadow:
                # Cast back to original dtype.
                out[k] = self.shadow[k].to(dtype=v.dtype, device="cpu").contiguous()
            else:
                out[k] = v.detach().to("cpu").contiguous()
        return out


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(f"zstd available: {_HAS_ZSTD}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(
        f"config: NUM_LAYERS={args.num_layers} MODEL_DIM={args.model_dim} "
        f"MLP_MULT={args.mlp_mult} TRAIN_SEQ_LEN={args.train_seq_len} "
        f"PARTIAL_ROPE_DIMS={args.partial_rope_dims} LN_SCALE={args.ln_scale} "
        f"EMA_ENABLED={args.ema_enabled} EMA_DECAY={args.ema_decay} "
        f"QUANT_BITS={args.quant_bits} GPTQ_LITE={args.gptq_lite} "
        f"USE_ZSTD={args.use_zstd} ZSTD_LEVEL={args.zstd_level} "
        f"SLIDING_EVAL={args.sliding_eval} SLIDING_STRIDE={args.sliding_stride}"
    )

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        partial_rope_dims=args.partial_rope_dims,
        ln_scale=args.ln_scale,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # Sanity check: print the actual ln_scale_buf values to confirm LN Scale is wired.
    if args.ln_scale and master_process:
        scales_seen = [float(b.ln_scale_buf.item()) for b in base_model.blocks]
        log0(f"ln_scale_buf values per layer: {scales_seen}", console=False)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # EMA must be initialized AFTER warmup state is restored (so it tracks real training).
    ema = EMAShadow(base_model, args.ema_decay) if args.ema_enabled else None
    if ema is not None:
        log0(f"EMA enabled: decay={args.ema_decay} warmup_steps={args.ema_warmup_steps}")

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # EMA update (after optimizer step so we capture post-update weights).
        if ema is not None and step >= args.ema_warmup_steps:
            ema.update(base_model)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size (raw): {model_bytes + code_bytes} bytes")

    # Decide what to quantize: live weights or EMA weights.
    if ema is not None and ema.steps > 0:
        log0(f"Using EMA weights for export (ema_steps={ema.steps})")
        export_state = ema.state_dict_for_swap(base_model)
    else:
        export_state = {k: v.detach().to("cpu").contiguous() for k, v in base_model.state_dict().items()}

    quant_obj, quant_stats = quantize_state_dict(export_state, args)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob, codec = compress_blob(quant_raw, args)
    quant_raw_bytes = len(quant_raw)
    quant_obj["__codec__"] = codec  # stored separately on disk too

    if master_process:
        artifact_path = "final_model.intq.ptz"
        with open(artifact_path, "wb") as f:
            # Write a tiny header: 4-byte codec name length + codec name + blob.
            cname = codec.encode("utf-8")
            f.write(len(cname).to_bytes(4, "little"))
            f.write(cname)
            f.write(quant_blob)
        artifact_bytes = os.path.getsize(artifact_path)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int_payload_bytes"], 1)
        log0(
            f"Serialized model {codec}: {artifact_bytes} bytes "
            f"(payload:{quant_stats['int_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size {codec}: {artifact_bytes + code_bytes} bytes")
        if artifact_bytes + code_bytes > 16_000_000:
            log0(f"!!! ARTIFACT OVER 16MB CAP !!! over by {artifact_bytes + code_bytes - 16_000_000} bytes")

    if distributed:
        dist.barrier()

    # Roundtrip: read back, decompress, dequantize, eval.
    artifact_path = "final_model.intq.ptz"
    with open(artifact_path, "rb") as f:
        cname_len = int.from_bytes(f.read(4), "little")
        cname = f.read(cname_len).decode("utf-8")
        blob_disk = f.read()
    quant_state_disk = torch.load(io.BytesIO(decompress_blob(blob_disk, cname)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict(quant_state_disk), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()

    # Use sliding eval for the final score if enabled.
    if args.sliding_eval:
        q_val_loss, q_val_bpb = eval_val_sliding(
            args, model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.sliding_stride,
        )
        eval_kind = f"sliding_s{args.sliding_stride}"
    else:
        q_val_loss, q_val_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        eval_kind = "disjoint"

    torch.cuda.synchronize()
    log0(
        f"final_intq_{cname}_roundtrip_{eval_kind} val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_intq_{cname}_roundtrip_{eval_kind}_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()