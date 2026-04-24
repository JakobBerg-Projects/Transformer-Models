# Part 2: Decoder-only Model for Text Generation
## Decoder Block
The decoder block is implemented similarly to the encoder block described in Part 1, with two sub-layers: a masked multi-head self-attention layer and an MLP block, each wrapped with a residual connection and layer normalization. The key difference from the encoder block is the addition of a causal mask in the attention layer, which prevents each token from attending to future positions, essential for autoregressive text generation. The MLP block follows the same structure as in Part 1, with two linear layers and a GELU activation with a hidden size of 4d.


## Model 

### Embedding Layer
Raw token IDs are integers. They carry no semantic meaning as numbers. The embedding layer is a learned lookup table that maps each token ID to a dense vector of dimension d. These vectors are updated during training, so the model learns to place semantically similar tokens nearby in this
d-dimensional space.

### Positional Encoding
Attention has no built-in sense of order. It treats its input as a set, not a sequence. Positional encoding fixes this by adding a position-dependent signal to each token embedding before it enters the decoder stack. The standard approach uses fixed sinusoidal functions of different frequencies across the embedding dimensions, so each position gets a unique pattern. This lets the model reason about which token came first, second, etc.

### Decoder Block (×N)
Each block has two sub-layers with residual (skip) connections and layer normalization around each:
1. #### Masked Multi-Head Self-Attention
This is where tokens attend to each other. Each token produces a query, key, and value vector. Attention scores are computed between all query-key pairs, softmaxed, and used to weight the values. The result for each token is a context-aware representation that blends information from other tokens.
Two masks control which tokens are allowed to attend to which:

Causal mask — enforces left-to-right generation. Token at position i can only attend to positions ≤ i. The mask is an upper-triangular boolean matrix filled with True (meaning "block this") above the diagonal. Without this, the model could cheat during training by looking ahead at the answer it's supposed to predict.
Key padding mask — prevents tokens from attending to [PAD] tokens. Padding is added to make all sequences in a batch the same length, but these are meaningless positions. The mask is True at padding positions so they are excluded from attention score computation.

Both masks are applied simultaneously inside MultiheadAttention via the attn_mask and key_padding_mask arguments respectively.
2. #### MLP Block (Feed-Forward Network)
A position-wise two-layer MLP applied independently to each token. It uses a hidden size of 4d and a GELU activation:
MLP(x)=Linear(GELU(Linear(x))) 
This is where most of the model's representational capacity lives. The attention layer routes information between tokens, but the MLP transforms it.
Dropout is applied after both sub-layers, and layer normalization is applied before each sub-layer (pre-norm style, as in GPT-2).

### Linear Classification Layer
After the NN
N decoder blocks, a single linear layer projects from d dimensions to vocabulary size V. The output at each position is a vector of logits, one per token in the vocabulary representing the unnormalized log-probabilities of what token comes next.


## BPE Tokenizer

The BPE tokenizer is trained on the full GooAQ dataset by iteratively merging the most frequent adjacent byte-level character pairs into new subword tokens, until the target vocabulary size of 20,000 tokens is reached. Tokens that appear fewer than 5 times are discarded via the min_frequency threshold. Compared to word-level tokenization, BPE handles rare and out-of-vocabulary words more gracefully. Rather than mapping an unknown word to a single [UNK] token, BPE decomposes it into known subword units. For example, an uncommon word like "unhappiness" might be split into "un", "happiness" rather than replaced wholesale with [UNK]. This results in a more informative representation of rare words and a more compact vocabulary than pure word-level tokenization, which would require a separate token for every unique word in the corpus.

## Training

The model was trained for 5 epochs on the full GooAQ subset of 859,765 question-answer pairs, with a batch size of 128 and a learning rate of 0.0001. The model has 36,261,920 parameters with an embedding size of 512, 8 attention heads, and 5 decoder layers.
The training loss decreased steadily across all epochs:

Epoch   Mean Cross-Entropy Loss
1       4.7823
2       3.8625
3       3.5208
4       3.3187
5       3.1840

The loss decreased consistently across all five epochs with no signs of plateauing, suggesting the model could have benefited from additional training epochs. The largest drop occurred between epoch 1 and 2 (~0.92), with the rate of improvement gradually slowing in later epochs, a typical pattern for cross-entropy loss in language model training. Notably, each epoch took roughly 10–11 minutes on GPU, for a total training time of around 52 minutes.
Since no validation set was used, it is difficult to determine whether the model began overfitting at any point. However, given that the training loss was still decreasing at epoch 5 without any sign of flattening, underfitting is more likely the dominant issue, the model simply needed more epochs rather than fewer.

## Challenges

The implementation process was largely straightforward. The dataset pipeline, decoder block, and sampling strategies were implemented without major issues, with only minor bugs encountered along the way, such as ensuring the [END] token was preserved after truncation and correctly shifting the source and target sequences by one position. The main challenge was the quality of the model outputs.  

## Prediction analysis

The model's predictions were generally poor. Generated answers rarely had any meaningful connection to the input question and often consisted of incoherent or repetitive text. This is likely a consequence of the model being relatively small (~36M parameters) and trained on a limited subset of the dataset for five epochs, which is insufficient for a language model of this complexity to generalize well.

Sampling strategy. Greedy sampling tended to produce repetitive, deterministic outputs, often getting stuck in loops where the same token or phrase was repeated until the maximum sequence length was reached. This is a well-known failure mode of greedy decoding. Top-p sampling produced more varied outputs due to the stochastic sampling from the nucleus, but the added randomness also introduced more incoherence. At higher temperatures the outputs degraded into near-random token sequences, while lower temperatures brought the behavior closer to greedy.

Overfitting. Given that training loss decreased quickly to around 3 but the model produced poor answers at inference time, there are signs of overfitting to surface-level token patterns rather than learning to model question-answer relationships. The dataset implementation in QADataset truncates long sequences and forces an [END] token at position max_length, which means many answers are cut off mid-sentence during training. This likely hurt the model's ability to learn coherent answer structure.

Hallucinations. The model frequently generated fluent-sounding but factually wrong or nonsensical answers, which is a classic hallucination pattern. Since the model has no retrieval mechanism and limited training, it has no reliable grounding in facts. It simply learns to produce token sequences that statistically resemble answers in the training data.

An example of these problems is when we asked the model what Bergen is and it replied "dova is a form of abnormar (alkali) nateur (alkali) (nate) with blood samples". This is clearly not what bergen is, and tells us that out model is not good enough to generalize to new questions. 

## How would we improve the model?

Given more time and resources, the most impactful improvement would be to replace the from-scratch training approach with fine-tuning a pretrained model such as GPT-2 or a similar publicly available language model. These models have already been trained on massive text corpora and have learned rich representations of language, meaning fine-tuning on the QA dataset would require far less data and compute to achieve coherent outputs.

Beyond that, several other improvements would likely help:

More data and longer training. The model was trained on a subset of the dataset for a few epochs. Training on the full dataset for multiple epochs with a learning rate schedule (such as cosine annealing with warmup) would give the model a much better chance of learning meaningful question-answer patterns.

Larger model. Increasing the number of decoder blocks, the embedding dimension, or the number of attention heads would give the model more representational capacity, at the cost of compute.

Better sampling. Implementing more sophisticated decoding strategies such as beam search or top-k combined with top-p sampling could improve output quality without any changes to the model itself.

Validation set and early stopping. The current setup has no validation set, making it difficult to detect overfitting during training. Adding one and using early stopping would help select the best checkpoint rather than simply the last one.

Longer sequence length. Many answers were truncated during training due to the max_length constraint, which likely hurt the model's ability to learn complete answer structure. More compute would allow a larger max_length.

