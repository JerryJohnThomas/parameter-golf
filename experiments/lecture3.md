# Lecture 3

(This is AI's helped take)

OpenAI baseline walkthrough from root `train_gpt.py`.

This is not a polished explanation yet. It is a first real code-reading pass of the
baseline, with emphasis on:
- what each section does
- why it matters for Parameter Golf
- what I still do not understand

## 1. Goal of the script

The root baseline `train_gpt.py` is both:
- the model definition
- the training loop
- the validation / BPB logic
- the post-training quantization + artifact measurement path

So unlike many repos, there is no separate `model.py` or `trainer.py`.

That matches the competition format:
- code size counts
- artifact size counts
- the script needs to be self-contained and reproducible

## 2. Hyperparameters

The `Hyperparameters` class defines most of the important knobs in the script.

### What it covers
- data paths
- tokenizer path
- run id / seed
- validation cadence
- total iterations
- warmup / warmdown
- batch tokens
- sequence length
- wallclock cap
- model shape
- optimizer split and learning rates

### Competition relevance
- `max_wallclock_seconds` is directly tied to the 10-minute challenge framing
- `vocab_size`, `num_layers`, `model_dim`, `mlp_mult`, `num_heads`, `num_kv_heads` affect:
  - BPB
  - speed
  - artifact size
- optimizer hyperparameters matter because the baseline uses a hybrid optimizer setup

### Important unknowns for me
- `qk_gain_init`
- `logit_softcap`
- why separate learning rates exist for:
  - `embed_lr`
  - `head_lr`
  - `tied_embed_lr`
  - `matrix_lr`
  - `scalar_lr`
- `tied_embed_init_std`
- `beta1`, `beta2`, `adam_eps` in this specific setup

## 3. Muon optimizer

The file defines:
- `zeropower_via_newtonschulz5`
- `Muon`

### What I understand
- Muon is used for matrix-shaped parameters in transformer blocks
- it is not used for everything
- the baseline later splits parameters into:
  - matrix params -> Muon
  - scalar / vector params -> Adam
  - embeddings / head -> Adam

### Competition relevance
- optimizer choice matters because training time is capped
- Muon seems to support aggressive learning on big 2D weights

### What I do not understand yet
- the actual Newton-Schulz orthogonalization logic
- why this update is better than Adam here in practical terms
- the exact role of:
  - `backend_steps`
  - momentum warmup
  - the scale correction inside Muon

For now, I understand the role of Muon in the training system better than its math.

## 4. Tokenizer-aware BPB setup

The file section calls this "tokenizer-agnostic evaluation setup".

Important functions:
- `build_sentencepiece_luts`
- `load_validation_tokens`
- `eval_val`

### What I understand
- raw token loss is not the final challenge score
- the challenge score is BPB
- because people can use different tokenizers, evaluation must map token predictions back to byte-level accounting
- `build_sentencepiece_luts` builds lookup tables for byte accounting and token boundary behavior

### Competition relevance
This is one of the most important challenge-specific parts of the baseline.
If BPB accounting is wrong, the score is meaningless.

### What I do not understand yet
- the exact logic of:
  - `base_bytes_lut`
  - `has_leading_space_lut`
  - `is_boundary_token_lut`
- the exact conversion from token losses to final BPB
- where the edge cases are for SentencePiece token boundaries

### Clarification to myself
The warning in the comments is not "never touch this function".
It means tokenizer-related scoring is sensitive, and bugs here could fake improvement.

## 5. Post-training quantization

This section includes:
- tensor sizing helpers
- int8 quantization
- dequantization
- compressed artifact serialization

### What I understand
- the baseline trains in bf16 / fp32
- that would be too large to submit directly
- so after training, the model is quantized to int8
- then compressed with zlib
- then measured as part of the artifact budget

### Competition relevance
This is core to Parameter Golf.
The trained model is not the final submitted representation.
The final artifact is the compressed exported form plus code bytes.

### What I do not understand yet
- exact tensor-by-tensor export policy
- which tensors remain float and which are quantized
- how the scales are stored
- how the round-trip validation works in detail

## 6. Data loading

This section includes:
- `load_data_shard`
- `TokenStream`
- `DistributedTokenLoader`

### What I understand
- shards are loaded from preprocessed `.bin` files
- token streaming is sequential and deterministic
- training does not appear to use random sampling in the usual dataloader sense

### Competition relevance
- data order and loader simplicity affect reproducibility
- distributed slicing matters for multi-GPU runs

### What I do not understand yet
- exactly how `DistributedTokenLoader` slices the shared stream across ranks
- how the extra `+1` token is used to build `(x, y)` pairs
- whether this loader design has any subtle effect on optimization quality

So this section is not just "standard dataloader stuff". It is part of how the training system stays simple and deterministic.

## 7. Transformer modules

This is the main model implementation area.

Important pieces:
- `RMSNorm`
- `CastedLinear`
- `restore_low_dim_params_to_fp32`
- `Rotary`
- `apply_rotary_emb`
- `CausalSelfAttention`
- `MLP`
- `Block`
- `GPT`

## 7.1 RMSNorm

### What I understand
- the baseline uses RMSNorm, not LayerNorm
- this fits the modern pre-norm style

### Why it matters
- normalization choice affects stability and training speed

## 7.2 CastedLinear

### What I understand
- this is not just a random linear layer wrapper
- it exists to control dtype behavior
- comment says: keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute

### Competition relevance
- this is part of the mixed-precision strategy
- it helps balance stability with speed

## 7.3 Rotary / RoPE

### What I understand
- this is the RoPE implementation
- it caches cos/sin tables by sequence length and device
- applies rotation to Q and K

### Competition relevance
- RoPE avoids learned positional embedding parameters
- that matters in a parameter-constrained competition

### What I still do not understand
- full math details of the implementation

## 7.4 CausalSelfAttention

### What I understand
- this is standard causal self-attention at a high level
- but in this baseline it is not fully generic textbook attention:
  - it uses grouped-query attention
  - it uses RoPE
  - it has query/key gain initialization via `qk_gain_init`

### Competition relevance
- attention structure affects:
  - parameter count
  - context handling
  - speed

## 7.5 MLP

### What I understand
- standard baseline feed-forward block
- uses relu^2 style nonlinearity
- MLP width is controlled by `mlp_mult`

### Competition relevance
- MLP size is one of the clearest parameter-budget levers

## 7.6 Block

### What I understand
- transformer block = norm + attention + norm + MLP + residual structure

## 7.7 GPT

### What I understand
- this assembles token embedding, transformer blocks, final norm/head behavior
- it supports tied embeddings
- it applies logit softcapping before cross-entropy

### What I still need to understand better
- exact role of `logit_softcap`
- exact placement and effect of `qk_gain_init`

## 8. Training section

The `main()` function is where everything is wired together.

### What I understand
- script expects CUDA
- supports distributed training through DDP
- uses world-size-aware grad accumulation with:
  - `grad_accum_steps = 8 // world_size`
- has a wallclock budget
- sets fast math knobs for CUDA attention backends
- seeds everything
- loads tokenizer and validation tokens
- builds model
- groups parameters into optimizer buckets
- runs training + validation + export

### Competition relevance
This is where the actual challenge engineering lives:
- speed constraints
- distributed assumptions
- optimizer grouping
- validation cadence
- final artifact evaluation

### What I do not understand yet
- the fast math backend choices in practical terms
- exact LR scheduling behavior in the loop
- full export / final validation flow after training

## 9. Serialized / round-trip validation

I originally marked this as "no idea", but it is important enough to be explicit.

### My current understanding
After training, the script does not just report validation on live fp/bf16 weights.
It also:
- quantizes and serializes the model
- compresses it
- measures the artifact size
- dequantizes / restores it
- validates the round-tripped model again

### Why this matters
That is very close to the actual competition object.
The challenge is not only "train a good model".
It is "train a model that still performs after being packed into a tiny artifact".

## 10. Current status of my understanding

### I now understand reasonably well
- why the file is self-contained
- the broad purpose of each major section
- why BPB matters more than raw token loss
- why quantization is central
- why tied embeddings and GQA matter in this competition
- why Muon exists in the baseline training recipe

### I do not understand well enough yet
- exact BPB byte-accounting mechanics
- exact quantization / round-trip mechanics
- distributed token slicing
- `qk_gain_init`
- `logit_softcap`
- the optimizer LR split rationale
- the practical effect of the CUDA backend knobs

## 11. Best next move

Before moving to Lecture 4, I should answer these baseline-specific questions:

1. Why are there separate `embed_lr`, `head_lr`, `tied_embed_lr`, `matrix_lr`, `scalar_lr`?
2. What exactly does `qk_gain_init` affect?
3. What exactly does `logit_softcap` do, and why might it help?
4. How does `DistributedTokenLoader` partition data across ranks?
5. What precision policy does `CastedLinear` implement?
6. How is BPB computed from token losses and tokenizer byte info?
7. What is exported in the quantized artifact, and how is it validated?
8. Which params go to Muon vs Adam, and why?
