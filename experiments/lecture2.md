# 

Lecture 2, Continuation of Component.md


from chatgpt 
  The missing parts, which matter much more for this challenge, are:

  - BPB scoring: token loss is not the final score; byte accounting matters.
  - artifact size: trained model precision is not what gets submitted; compressed quantized weights matter.
  - GQA: num_heads=8, num_kv_heads=4, so K/V are shared across groups of query heads.
  - Muon: most matrix weights are trained with Muon, not Adam.
  - BF16/FP32 casting: weights and compute precision are deliberately mixed.
  - tied embeddings: input embedding and output head can share weights, saving many params.
  - residual stream: this is the central “state” passed through blocks.
  - post-training quantization: int8 baseline, later records use more aggressive methods.
  - training loop constraints: 10-minute wallclock is a first-class design constraint.
  - evaluation tricks: sliding eval / TTT in records are not ordinary training.
  -  Test time inference

## Legend
* any open questions i mark it with doubt
* once answered i rename doubt to question 
* as an LLM please look at all open questions with doubt tag

# Tokenizer
* Smaller Vocabulary means more the number of tokens you need for one word .

Parameter Size
* Every token in your vocabulary needs its own Embedding Vector.
* If your hidden dimension is 768 and your vocabulary is 50,000, then $50,000 \times 768 \approx 38.4$ million parameters.
* In Parameter Golf, 38M parameters is roughly 150MB (at float32). You would fail the 16MB limit instantly.

question: so big tokenizer iwth big vocabular and small vocabulary is the differnese in the output dimensino or the hidden dimension which is the output dimension
Answer:
* Neither — it's the vocab dimension specifically.
* The embedding table shape is - `[vocab_size × hidden_dim]`
* Hidden dim (512) stays fixed regardless of vocab size. What changes is vocab_size:
* So bigger vocab = bigger embedding table = more params = less room for transformer layers in 16MB.


Efficiency Gain
* On the flip side, a larger vocabulary makes the model faster at processing text.
* If a sentence is 10 tokens instead of 50 tokens, the Self-Attention mechanism (which is $O(n^2)$) has $5^2$ ($25\times$) less work to do
* In your 10-minute training window, a model with a larger vocab can "see" much more text (more tokens of information) in the same amount of time.

# BPB - Bits Per Byte
* in ML world, cross entropy loss = -log(p_correct)
* Here in Parameter Golf Competition - Raw token loss is not comparable across different tokenizers.

Formula

$$\text{BPB} = \left( \frac{\text{Average Token Loss}}{\ln(2)} \right) \times \left( \frac{\text{Total Tokens}}{\text{Total Bytes}} \right)$$

Concrete ExampleLet's use your "transformer" example. The word "transformer" has exactly 11 characters (11 bytes).
Model A (Small Vocab)
Tokens (3): trans + form + er
Token Loss: Let's say the model predicts each chunk pretty easily, getting an average loss of 2.0 per token.
Total Word Loss: $3 \text{ tokens} \times 2.0 = 6.0 \text{ nats}$
BPB: $\frac{6.0}{\ln(2) \times 11} \approx \frac{8.65 \text{ bits}}{11 \text{ bytes}} \approx \mathbf{0.78 \text{ BPB}}$

Model B (Large Vocab)
Tokens (1): transformer
Token Loss: Predicting the entire big word at once is harder! Let's say the loss for this single token is 5.0.
Total Word Loss: $1 \text{ token} \times 5.0 = 5.0 \text{ nats}$
BPB: $\frac{5.0}{\ln(2) \times 11} \approx \frac{7.21 \text{ bits}}{11 \text{ bytes}} \approx \mathbf{0.65 \text{ BPB}}$The Verdict: If you only looked at the raw PyTorch loss output, Model A (2.0) looks vastly superior to Model B (5.0). But when normalized to BPB, Model B (0.65) is actually the better compressor and understander of the text than Model A (0.78).


SO this is a form of a loss, the lower the better, if we are very confident then the p_score will be close to 1 and the log of that will be closer to 0. 
So the competition is to have this as the best metric

# Artifact Size and Submission (Golf Specific)
The size is always deterministic

```
You run train_gpt.py on 1×H100
→ trains model
→ at end: quantizes weights, compresses, measures size
→ prints: val_bpb = 1.21, artifact = 15.9MB
→ saves train.log

You submit PR with:
→ train_gpt.py (your modified version)
→ train.log (proof of your score)
→ README.md
→ submission.json

OpenAI re-runs your train_gpt.py on 8×H100
→ verifies val_bpb matches
→ verifies artifact < 16MB
→ accepts to leaderboard
```

# KV Cache

This is something that happens at inference 

## Problem
When a model generates a sentence, it predicts one word at a time (autoregressively).Let's say it has already generated 1,000 words and is predicting word 1,001.

To calculate the Attention scores for word 1,001, the Query ($Q$) for that new word needs to do a dot product with the Keys ($K$) of all 1,000 previous words. Instead of recalculating those 1,000 $K$ and $V$ vectors from scratch every single step, the model saves them in your GPU memory. This saved memory bank is called the KV Cache.



# A practical Example of Attention with KV Cache

We have 1000 words and we are trying to predict the 1001 word 

Step 1: Compute the New Vectors
* When word 1001 enters the Attention layer, the model calculates three brand-new vectors for it: $Q_{1001}$, $K_{1001}$, and $V_{1001}$.

Step 2: The Query Asks the Questions
* Word 1001 needs to figure out its context. It uses its Query ($Q_{1001}$) to "ask" all the previous words: "Hey, I am word 1001. Which of you previous words are mathematically relevant to me right now?"
* To get the answer, $Q_{1001}$ is multiplied (dot product) against the Keys ($K$) of all 1,000 previous words, plus its own Key.
$$\text{Scores} = Q_{1001} \cdot [K_1, K_2, K_3, ..., K_{1000}, K_{1001}]$$

Step 3:
* Output Computation with the attention scores multiplied with V


Step 4: Update the Cache and Throw $Q$ Away
* Once word 1001 is done gathering its context, its job as the "asker" is over. We completely throw away $Q_{1001}$.
* However, we save $K_{1001}$ and $V_{1001}$ into the KV Cache.Why? Because in the very next millisecond, word 1002 is going to enter the system. 
* Word 1002 will generate its own Query ($Q_{1002}$), and it will need to search the Keys of words 1 through 1001.



# KV Cache solution


The thing we saw earlier was the standard - **Multi-Head Attention (MHA)**

To solve this problem we introduced two things
* Multi-Query Attention (MQA) - The Extreme Squeeze
  * Share one set of Keys and Values across all the Query heads.
  * Architecture: 32 $Q$ heads, but only 1 $K$ head and 1 $V$ head.
  * Result: The KV cache size shrinks by $32\times 1$ Inference becomes incredibly fast.
  * problem: we loose a lot of variation theoretically giving the same answer to different kinds of queries.


* Grouped-Query Attention (GQA) - The Goldilocks Zone
  * GQA is the elegant middle ground introduced around 2023, and it is what powers almost all top-tier modern models like LLaMA 3, Mistral, and Gemma.
  * Instead of forcing all queries to share one Key/Value head, we divide the Query heads into groups
  * Architecture: If you have 32 $Q$ heads, you might create 8 groups. 
  * Each group gets its own $K$ and $V$ head.Ratio: 4 $Q$ heads share 1 $K$ head and 1 $V$ head.
  * Result: You still slash your KV Cache memory footprint (in this case, by an $8\times$ reduction compared to MHA), but because the queries are grouped logically, the model retains almost all of the performance and reasoning quality of standard MHA.


Question: how does inference have anything to do with model training weight size
Answer
* KV cache is purely an inference optimization. It has zero effect on training or model weight size.
* The reason GQA appears in both contexts is:
  * At inference: GQA reduces KV cache memory (fewer K/V vectors to store per token)
  * At training/parameter golf: GQA reduces W_K and W_V parameter count (smaller matrices to store in 16MB)
* [GOLF] Same architectural change, two separate benefits. In this competition we care about the second one — parameter savings — not the inference speed benefit.

# GQA — Grouped Query Attention

So normally we know there are 8 independent Q K V vectors right

## Motivation

* Researchers noticed that K and V heads learn very similar things across groups of Q heads. 
* Heads 0 and 1 tend to have nearly identical K and V patterns. So you're spending parameters on redundancy
* What if multiple Q heads shared the same K and V?

so the heads are grouped in pairs

num_heads    = 8   (Q heads)
num_kv_heads = 4   (K and V heads)


```
Q head 0 ─┐
Q head 1 ─┴─→ shares K[0], V[0]

Q head 2 ─┐
Q head 3 ─┴─→ shares K[1], V[1]

Q head 4 ─┐
Q head 5 ─┴─→ shares K[2], V[2]

Q head 6 ─┐
Q head 7 ─┴─→ shares K[3], V[3]
```


Question: we are only thinking about decreasing the number K and V< why not think about decreasing Q heads here?
Answer
* The Query ($Q$) is the "Asker." The number of $Q$ heads determines how many different, complex questions the current token can ask the context at the exact same time.
  * For the word "apple":
  * Head 1 might ask: "Is this the fruit or the tech company?"
  * Head 2 might ask: "Is this the subject or the object of the sentence?"
  * Head 3 might ask: "What color is being described?"
* The Keys and Values ($K, V$) are the "Database."
  *  Researchers found that you don't actually need 32 separate databases to answer 32 different questions. Several $Q$ heads can query the exact same $K/V$ database and still extract the specific information they need.
* Or another analogy, Think of it like a library:
```
Q = different readers asking different questions
K = the card catalogue (can be shared)
V = the actual books (can be shared)
```
* The Golden Rule of GQA: It is much better for a model to be incredibly inquisitive (many $Q$ heads) while searching a compressed database (few $K/V$ heads), than to be a simple-minded model (few $Q$ heads) searching a massive, highly-detailed database.





