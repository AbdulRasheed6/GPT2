## 1. The Big Picture: A Decoder-Only Architecture

The model is a **Generative Pre-trained Transformer (GPT) style decoder**. Unlike the original Transformer, which was built for translation (Encoder + Decoder), this model is a **Next-Token Predictor**. We discard the Encoder and Cross-Attention, leaving a stack of "Identical Blocks" that process the sequence.

### Structural Components of a Block

Each block in our model performs two main tasks: **Communication** (Attention) and **Computation** (Feed-Forward).

1. **Input & Positional Embeddings:** Since Transformers have no recurrence (like RNNs), they don't know the order of words. We add **Positional Encodings** to the **Token Embeddings** so the model knows which token is at index 0 vs. index 255.
2. **Masked Multi-Head Self-Attention:** This is the "communication" layer. Masking ensures the model is **autoregressive**—it cannot "cheat" by looking at the word it is trying to predict.
3. **Feed-Forward Network (FFN):** If Attention is communication, the FFN is "thinking." This is a linear layer followed by a non-linearity (like ReLU or GeLU). It acts as a **knowledge bank**, processing the information gathered during the attention phase.
4. **Residual (Skip) Connections:** We add the input of the block back to its output (). This allows gradients to flow through the network during backpropagation without vanishing, enabling us to train deeper models.
5. **Layer Normalization:** We normalize the features to have a mean of 0 and a variance of 1. This stabilizes the training process and prevents internal covariate shift.
6. **Dropout:** A regularization technique where we randomly "shut off" some neurons during training. This forces the model to be robust and prevents it from over-relying on specific patterns (overfitting).

---

## 2. Data Dimensions & Tensors

We handle data in 4D or 3D tensors. For our implementation, the core shape is :

* **B (Batch):** 64 (Processing 64 sequences at once).
* **T (Time/Block Size):** 256 (The maximum context length/history the model sees).
* **C (Channel/Embed Dim):** `n_embed` (The size of the vector representing a single token).

---

## 3. The Mechanics of Masked Self-Attention

Self-attention allows tokens to "vote" on which other tokens are relevant to them.

### The Mathematical Flow

We derive three vectors for every token using trainable weights :

1. **Query ():** What am I looking for?
2. **Key ():** What information do I contain?
3. **Value ():** If I am relevant, what information do I contribute?

**The Score Calculation:**
attention_weights= Softmax((Q@K.T) * (head_size)**-0.5) * V


* **Matrix Multiplication ():** This creates an "Affinity Matrix" of size , showing how much every token in the sequence relates to every other token.
* **Scaling:** We divide by  to keep the values small, ensuring the Softmax gradients stay healthy.
* **The Mask (Tril):** We apply a lower-triangular mask (`torch.tril`). This fills the upper triangle (the "future") with .
* **Softmax:** The  values become , and the remaining values are normalized to sum to 1. This creates a "weighted map" of the past.
* **Value Aggregation:** We multiply this map by  to get the final context-aware representation of the token.

---

## 4. Scaling to Multi-Head

Instead of one large attention calculation, we use **Multi-Head Attention**. We split the `n_embed` into several smaller "heads" (e.g., 8 heads of 32 dimensions each).

* Each head looks for different patterns (one might look for grammar, another for logic).
* **Projection:** After the heads run in parallel, we concatenate them and pass them through a final linear layer (Projection) to blend their findings.

---

## 5. The Training Loop

1. **Forward Pass:** Data goes through the Embeddings  Blocks  LayerNorm  Linear Head.
2. **Loss:** We calculate the **Cross-Entropy Loss** between the predicted logits and the actual next token in the sequence.
3. **Backprop:** We update the weights to minimize the loss.
4. **Generation:** To see it work, we pass a starting token, take the last logit, apply Softmax to get a probability, and "sample" the next token. We append that token to the input and repeat.

