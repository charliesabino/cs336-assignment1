transformer_accounting:

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

For 2 * 1,555,124,800 = 3,110,249,600 = ~3 GB total
