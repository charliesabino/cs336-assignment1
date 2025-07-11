transformer_accounting:

---

a.
Consider GPT-2 XL, which has the following configuration:

vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400

Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?

---

Answer:
Embedding/Unembedding table \in [vocab_size, d_model] -> [50257, 1,600] = 80,411,200

Each head:
Q, K, V \in [d_model, d_k] -> [1,600, 1,600 / 25 = 64] = 102,400

Each block:
The attention heads gives us a total of 25 *3* 102,400 = 7,680,000 parameters per block

We have one output projection matrix \in [d_model, d_model] -> [1,600, 1,600] = 2,560,000

We have two RMS norms \in [d_model] -> 2 * 1600 = 3200

For ffn, we have three matrices (assuming swiglu) of size [1,600, 6,400] (one having its dimensions flipped)
= 3 * 10,240,000 = 30,720,000

o.w. we have 2 -> 2 * 10,240,000 = 20,480,000

For a total of 7,680,000 + 2,560,000 + 3200 + 20,480,000 = 30,723,200 parameters per block

Giving us 48 * 30,723,200 = 1,474,713,600 parameters for the transformer blocks

So we have 80,411,200 + 1,474,713,600 = 1,555,124,800 parameters total

For 4 * 1,555,124,800 = 3,110,249,600 = ~4 GB total

---

b. Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length tokens

---

Answer:

To obtain key, query, and value vectors:
|{Q,K,V}| * context_len * 2mnp = 3 * 1024 * 2 * 1600 * 1 * 64 = 629,145,600

Then, we dot product each query vector and each key vector:
64 * 2 * 1024^2 = 134,217,728

Then scale each value vector by the attention weight:
1024 * 64 = 65,536

And add it back to each embedding:
1024 * 1600 = 1,638,400

We do this across 25 heads:
25 * (629,145,600 + 134,217,728 + 65,536 + 1,638,400) = 19,126,681,600

Then, for FFN, we do two matmuls for 2 * 2 * 1600 * 6400 * 1 = 40,960,000

And for the output matrix we have 2 * 1024 * 1600 * 1600 = 5,120,000

And we have
19,126,681,600 + 40,960,000 + 5,120,000 = 19,172,761,600 FLOPs per block

Across 48 blocks: 48 * 19,172,761,600 = 920,292,556,800

And one final linear layer: 2mnp = 2 * 1600 * 
