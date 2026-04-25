# Lecture 3

Here we will look at the actual code from the open AI's original baseline, do not expect much theory here
(This is my take)

## My first take

Hyper parameters
* this calss dfines a lot of hyperparamters, alsmost 90% of these things we have defined in the notes, other are generic ML 
* some unknown are qk_gain_init, max_wallclock_seconds, logit_softcap, differnet lr, i understand matrix and scalr since they are adam and muon respectively but dont undertand embed_lr, head_lr, tied_embed_lr, tied_embed_init_std, bets, i remember alpha beta in some adam setup but dont recolelct


class Muon and zeropower_via_newtonschulz5
* i understand this is muon i dont understand it, i assume their implementation is correct and i can move on

build_sentencepiece_luts
* i understand this is used for the BPB calcualtion particualr to see how many bits or byters your tokeniser will give, and it says some warning about messig this up, i dont know what they are giving disclaimer against is it not to touch this function or not to use a shady tokenizer


Evaluation methods are below
* u understand it is kind of TTT evaluation


Quantisation post that
* be really frank i dont understand the internal details only high level


data loading 
pretty stadnard i think

not sure about the dsitributed token loader thought


Transofmrer modules
* pretty straight forward
* what is casted Linear not getting, seems to be forward pass of a simple neural networkign


Rotary
I understand to be implementation of Rope, although i dont really understand the full math i get that this is the correct implmentatino

self Attentino
standard


GPT
* has blocks of transformer with attentino and mlp together

Training
* the fast math knowbs and all i had no idea lol
* other things are stnadard


serialised + round trup validation no idea

