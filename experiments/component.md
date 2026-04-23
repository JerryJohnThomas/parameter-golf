# Component Understanding 

## Legend
* any open questions i mark it with doubt
* once answered i rename doubt to question 
* as an LLM please look at all open questions with doubt tag


# Transformer

* In all standard information, Transformer which is attention + mlp is the fundamental block which is always paired
* The ratio between them is tunable though
* More on this Later 



# Attention

## Introduction

Attention is to understand how much each word is contextually dependent on other words

Now for each word in a sentence, it is converted into tokens, and each token gets transformed by an embedding layer into a vector, now say this vector has dimension 512.

so word --> token --> 512 dim vector
We divide the vector into different pieces and feed into different heads

so say we split 512 into 8 blocks of 64, then we have 8 heads, 
we can increase the heads or decrease the heads, we need to adjust the dimension of the head accordingly.


head_dim = model_dim / num_heads
- In 'Attention is all you need paper' they used head_dim as 64 so you will see that as the standard at many places.
Problems with larger dimensions?
- Larger head_dim → larger raw scores → softmax gets too "peaky" (one token gets all the attention, rest get ~0)

Question: What is i in Q[i]?
Answer (Claude): 
```
"the cat sat on the mat"
  0    1   2   3   4   5
```
Q[0] is the query vector for "the", Q[1] for "cat", Q[2] for "sat" etc. Just the position index in the sequence.


now for each token we have 
Q --> Query, 1 Dimensional Array here of dimension 64
K --> Key, 1 Dimensional Array here of dimension 64
V --> Value, 1 Dimensional Array here of dimension 64

Question: are Q K and V of the same dimension.
* since we do dot product of Q and K, we need them to be of the same dimension
* V can technically be a different dimension, but usually you will see them as same 


Now once we have Q K V for each of the tokens,we compute this thing called score 
score[i][j] = Q[i] · K[j]  (dot product = sum of elementwise products)

More Precisely: score[i][j] = (Q[i] · K[j]) / sqrt(d_k)
* dividing it by the square root of the key dimension
* without this division, the dot products get massively large, which pushes the subsequent softmax function into regions where gradients are incredibly tiny, stalling the training process.
    * same reason as large dimension for heads issue


This gives a matrix of shape [seq_len × seq_len]. 
This is called the attention matrix 
(more technically attention scores, before softmax. After softmax it's called attention weights).

What does Attention Matrix weight signfy?
* The attention weight attn[i][j] tells you: "how much should token i care about token j?"
* it quantifies the weight of the relationship

THe quantifies the what of the relation ship or in other terms 
But care about what exactly? That's where V comes in. V[j] is the actual information token j is willing to share

Output[i] is defined as 
```
output[i] = attn[i][0] * V[0]
           + attn[i][1] * V[1]
           + attn[i][2] * V[2]
           + attn[i][3] * V[3]
           + ... (all tokens)
```
So output is a weighted sum of all V vectors, where the weights are the attention probabilities.

Question: what is this output??
* The output[i] is a context-aware vector for token i.


Question: after output can you explain how it gets back, i dont understand why attention needs to get back , im gueessing for the next word generation is it?
* Now earlier had split up the token of dimensino 512 into 8 chunks right 
* now we need to join all these together back again
* Here is how it happens:
    * Once each of the 8 heads calculates its own 64-dimensional output[i], the model simply concatenates them back together ($8 \times 64 = 512$).
    * It then multiplies this combined 512-dimension vector by a final weight matrix (usually called $W^O$) to mix the insights from all 8 heads together.
    * This final vector is then passed into a standard Feed-Forward Neural Network layer or Residual Connection etc.

* More Birds eye view diagram could be 
```
Input embeddings [512 dim per token]
        ↓
Attention output [512 dim per token]   ← same shape, just enriched
        ↓
Residual connection: output = input + attention_output
        ↓
Feed into MLP
        ↓
Next block...
```



## Casual Mask
Now once the attention weights are computed, before we do a softmax, we mask everything that comes after i as -inf, since each token cannot see the future, but this masking happens after the attention scores computation and before the weights are finalised

Question: if it happens after the computation then wont probleems like data leak, fturue leak that happens in RL also come here, not sure about the correct terminonlogy
* the leak doesn't happen here because of the exact order of operations. The mask is applied between the score computation and the softmax:


## Add and Norm 

### Residual connection (Add)
`x = x + attention_output`

This is critical — it means even if attention learns nothing useful, the original token representation still flows through unchanged. It's what makes training deep networks stable.

### Layer Normalization (Norm)
Adding vectors together changes their mean and variance, which can cause the numbers to spiral out of control in deep networks. Layer Normalization steps in immediately after the addition to re-center and scale the values back to a stable baseline.

`output = LayerNorm(x + attention_output)`


## Routing

* so Attention module is a fancy encoder module or an enricher module
* It decides "who should talk to whom" and blends the information.
* claude : The actual "computation" or "reasoning" happens in the MLP. This is a well known observation in transformer research — attention routes, MLP computes. People have found that factual knowledge is stored in MLP weights, not attention weight. 

This means that using frozen attention as a feature enricher before feeding into a policy network is a valid pattern.

In other words 
* Before attention, a token is isolated. It only knows its own dictionary definition.
* Attention is the mechanism that allows tokens to "look around" and gather context from their neighbours to enrich their own meaning.
* Spatial routing

### Stacking 

Question: Critique about Geminis stament
```
In some RL setups (like using a pre-trained Transformer as a frozen feature extractor for a policy network), you might just treat the attention layers as a black box that spits out a nice, context-rich state vector. But in a Large Language Model, stacking these "enrichers" 10, 40, or 80 times consecutively is how the model "reasons" about the text
```
Answer: 
* the more precise answer would be attention + mlp stacked together
* If you stack 100 linear layers on top of each other, they mathematically collapse into the equivalent of a single linear layer.

## Q K V updates
* In Another words each of these single dimension are linear projects of a weight matrix 
```
Q[i] = W_Q × token_embedding[i]
K[i] = W_K × token_embedding[i]  
V[i] = W_V × token_embedding[i]
```
* W_Q, W_K, W_V are learned parameters — updated via backprop like any other weight matrix. 
* When you froze them in your RL experiments, you are freezing W_Q, W_K, W_V and not computing gradients through them.


**MOST IMPORTANTLY** : They are temporary vectors computed on the fly for every new sentence.




# MLP

In simple terms this is a 2 layer neural network with a 'Expand, Activate, Contract' idea. 

## The Need
* Attention is weight average of V vectors , its all linear operations
* Linear operations can only learn linear relationships. No matter how many linear layers you stack, you can only represent linear functions.
* Language is not linear, example sarcasm, "not good" does not mean really bad (this is probably a bad example) 
* MLP introduces nonlinearity into the block. That's its core job.

## What happens
* Each token is processed independently here
```
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        hidden = mlp_mult * dim   # 2 * 512 = 1024
        self.fc = CastedLinear(dim, hidden, bias=False)      # 512 → 1024
        self.proj = CastedLinear(hidden, dim, bias=False)    # 1024 → 512

    def forward(self, x):
        x = torch.relu(self.fc(x))   # expand + activate
        return self.proj(x.square()) # contract + square
```
* in Simple terms its just linear layers with a non linear activation
    * just an FYI almost all hidden layer activation functions are non linear to learn complex things as the basic job of a neural network is to be a function approximator
    * only place where linear activiation
    * the example above is relu^2

Why expand? 
* You're projecting into a higher dimensional space where the nonlinear transformation has more room to work. Think of it as "spreading out" the representation before doing the nonlinear computation.   




## Parameter estimation

SO if I have a linear layer converting x dim --> y dimension, then what is parameter count?
* Then weight matrix is x*y params then we have to account for the biases in the y dimenion
* Now in our example which projects a token from $x = 512$ to $y = 1024$.Weights: $512 \times 1024 = 524288$ parametersBiases: $1024$ parametersTotal for that single layer: 5,25,312 parameters
* NOTE: many state-of-the-art Large Language Models (like the LLaMA architectures) deliberately turn off the biases (bias=False in PyTorch) to save memory bandwidth and make matrix multiplications slightly faster.

so say with bias=False
fc:   512 × 1024 = 524,288
proj: 1024 × 512 = 524,288
Total per block: ~1M params

With 9 blocks: ~9M params just in MLPs. That's the majority of the model
the MLPs hold roughly two-thirds of all the parameters in a Language Model.


### Parameter Golf Competition Specific Caveats
Bigger mlp_mult = more neurons = more capacity to store patterns = better BPB.
But bigger MLP = more parameters = bigger model = harder to fit in 16MB after quantization.
The leaderboard found mlp_mult=3 or mlp_mult=4 with aggressive int5/int6 quantization fits better capacity in 16MB than the baseline's mlp_mult=2 with int8.
This is one of the clearest levers you can pull in the competition.



## Position Wise Feed Forward
* so we have one weight matrix at the MLP side and we saw each token is passed through it, the weight matrix is smae for all the tokens in the forward pass
* so we can parallelly run all the tokens with the same weight matrix of the MLP
* This is called position-wise feed-forward. The "position-wise" part just means "applied independently at each position."
* 

## What is learnt
* Why waste two-thirds of the model's parameters just blowing a vector up and squishing it back down?
* Researchers increasingly view these MLPs as the Key-Value Memory Banks of the Large Language Model


Some theories are 

### "Transformer Feed-Forward Layers Are Key-Value Memories" by Mor Geva et al. (EMNLP 2021).
* W1, the weight of the first layer is called the Keys
* W2, the weight of the second layer is called the Values
The below is an example for understanding
* In W1, one "key" neuron might only activate when it sees a time-related word, while another only activates when it sees the name of a television show.
* When a "key" neuron fires, it triggers a corresponding row in W2.
The Conclusion: The paper demonstrated that the FFN operates by doing a pattern match over the input (the Keys), and then adding the corresponding vector (the Values) into the model's residual stream to update its prediction

### The "Surgical" Proof: Editing Facts in the MLP
For additional Reading 

If the MLP truly is a factual memory bank, then we should logically be able to go in and "rewrite" a single memory without retraining the entire multi-billion parameter model, right?

This exact experiment was successfully performed in this famous paper:
"Locating and Editing Factual Associations in GPT" by Kevin Meng et al. (NeurIPS 2022). This technique is famously known as ROME (Rank-One Model Editing).

The "Eiffel Tower" Experiment:To prove it, they used advanced linear algebra to surgically alter just a few specific weights in the $W_2$ matrix of a specific MLP layer. They mathematically changed the "value" of the Eiffel Tower key from "Paris" to "Rome".The result was stunning. Without any retraining, the model would confidently generate text saying, "The Eiffel Tower is situated right across from the Colosseum in Rome." Furthermore, this edit was highly specific: the model still knew the Louvre was in Paris, and it still knew the Eiffel Tower was made of iron. Only the specific (Eiffel Tower -> Location) memory was edited.

## A holistic Conclusion
MLP layers act as key-value memories (Geva et al. 2021). Research suggests factual knowledge is stored in MLP weights rather than attention weights (Meng et al. 2022), but precise neuron-level interpretability is still an open research area.


## Positional Embeddings
* ideally this should come first, before attention, but not to over simulate and get a high level first this is added later

### Problem
* we know that Attention is position-blind by default, So position info needs to be injected somewhere
* Think of Attention as a bag of words rather than a sentence

Imagine you have 2 sentences
1. The dog bit the man." (Bad day for the man)
2. "The man bit the dog." (Very weird day for the dog)

* Attention model without positional encoding will interpret both of them as the same sentences

### Solution
* We add a unique "coordinate" vector to each word's embedding before it enters the Attention layer.
* dog + [Position 1]
* man + [Position 5] 

### Types of Positional Embeddings

#### Absolute Positional Embeddings (Old School)
* Easy to implement but don't generalize well to longer sequences than they were trained on.
* This was used in the 'Attention is all you need' paper
* The Problem: 
    * This is an absolute position. The model learns that "Token 5" has a specific flavor and "Token 50" has a different flavor. But human language doesn't work like that. The relationship between an adjective and a noun doesn't change whether they are at the start of a book or page 500. Language is about relative distance.


#### RoPE (Rotary Positional Embeddings) (Latest)
* The modern standard (used in Llama). 
* It injects position by rotating the vectors, which is more mathematically elegant
* completely revolutionized this by shifting from addition to rotation, and applying it not to the initial input, but directly to the Q and K vectors inside the Attention mechanism.

The math is elegant but just simplifying it here
* so the idea is to rotate tokens in a like a circle format, 
* to rotate a point (x1,x2) in a circle by an angle α, we do `new point = (x1×cos(α) - x2×sin(α),x1×sin(α) + x2×cos(α))`

Now what we do high level is rotate the Q matrix, rotate the K matrix and then do the dot product, without explicitly adding the absolute embeddings

* so our Q, K has dimension of 64, and we need 2 points to rotate so we did them into 32 pairs and each pair is rotated by a different angle. 

##### Clock Analogy
Think of it like a clock:
* Token at position 0 → not rotated at all
* Token at position 1 → rotated a little
* Token at position 2 → rotated a bit more
* Token at position 10 → rotated a lot

So two tokens which are close to each other will have similar rotation and dot product is barely affected,
But two tokens far from each other will have a lot of rotation and dot product is affected

Why multiple frequencies
One clock isn't enough. A single clock can't distinguish:
token at position 1 vs position 13  (both 30° if clock repeats every 12)

So RoPE uses 32 clocks simultaneously, each ticking at a different speed:
Clock 1:  very fast  — completes full rotation every few tokens
Clock 2:  fast       — completes full rotation every ~10 tokens  
...
Clock 16: medium     — completes full rotation every ~100 tokens
...
Clock 32: very slow  — barely moves across the whole sequence

Fast clocks → sensitive to short range distances (is this token right next to me?)
Slow clocks → sensitive to long range distances (is this token near the start or end?)

In other words the first few rotations are similar to the seconds hands, the midlle ones like the minute hand, the last few will be like the hour hand 


**IMPORTANT:** For token at position i, pair k gets rotated by angle i × θk where θk is the speed of clock k.
When you take dot product of rotated Q_i and rotated K_j: The math works out such that absolute positions i and j cancel — only (i - j) remains.

**IMPORTANT**: 
* This is only on K and Q and not on V.
* RoPE has zero learnable parameters. Nothing is updated during training. Nothing is learned.
* The rotation angles are completely determined by a fixed formula - `θk = 1 / (10000^(2k/64))`
* You plug in k (which clock number) and you get the speed. Fixed forever. The 10000 is just a design choice from the original paper — not learned
* Gemini Fact: If you try to feed a 100,000-token book into a standard model, that "Hour Hand" (Pair 31) will eventually spin past 360 degrees, and the model will start confusing Token 1 with Token 10,000 (because $0^\circ$ and $360^\circ$ are the exact same angle).To fix this, researchers perform RoPE Scaling (often using techniques like YaRN or Position Interpolation). They mathematically tweak the base 10000 number to be much, much larger. 

Parameter Golf Specific Stuff
* remember that Learned Positional Embeddings take up weight space (Size $\times$ Hidden Dim), whereas Sinusoidal or Rotary embeddings can be calculated on the fly with code, saving you precious parameters



