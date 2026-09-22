import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1) # Flips the last two dimensions becoming [1, 64, 9] and computes against [1, 9, 64]
        d_k = K.shape[-1]
        scores = scores / (d_k ** 0.5) 

        weights = torch.softmax(scores, dim=-1) # Tells softmax to operate in the last dimension (9) and to turn those 9 raw scores into clean weights from 0 to 1
        output = weights @ V # Weighted average of all those 9 values

        return output, weights

# Testing it with real numbers now, SINGLE HEAD TEST

torch.manual_seed(0)

batch_size = 1
seq_length = 4
embed_dim = 8

x = torch.rand(batch_size, seq_length, embed_dim) # Creates a tensor shape [1, 4, 8], filled with random numbers from 0 to 1

attn = SelfAttention(embed_dim) # Instance from out class telling it to expect vectors of size 8, internally creating the 3 linear layers
output, weights = attn(x) 

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Weights shape:", weights.shape)
print("\nAttention weights for word 1:")
print(weights[0, 0]) # First sentence, first row of weights, SINGLE HEAD PRINTS

class MultiHeadAttention(nn.Module):
    def __init__(self, emed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5) # Flips last two dimensions of K then does matrix multiplication on each
        weights = torch.softmax(scores, dim=-1)
        attn_output = weights @ V # Another matrix multiplication so or every head eery word has a new contet-aware vector size 

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim) # Swap the order back with transpose, then merge the last two dimensions back into one deimension of size (embed_dim) aka the concatenation step of all 4 heads
        output = self.out_proj(attn_output)

        return output, weights

# Testing it with real numbers, MULTI-HEAD TESTS

print("\n--- Multi-Head Attention ---")

num_heads = 4
mha = MultiHeadAttention(embed_dim, num_heads)
mh_output, mh_weights = mha(x)

print("Output shape:", mh_output.shape)
print("Weights shape:", mh_weights.shape)
print("\nHead 1's attention weights for word 1:")
print(mh_weights[0, 0, 0])
print("\nHead 2's attention weights for word 1:")
print(mh_weights[0, 1, 0]) # MULTI-HEAD PRINTS       