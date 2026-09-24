# Donwloads Shakespeare's works as plain text and previews it for training data for a language model practice

import torch
import torch.nn as nn
import urllib.request

from attention import TransformerBlock

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "shakespeare.txt")

with open("shakespeare.txt", "r") as f:
    text = f.read()

print("Total characters:", len(text))
print("First 200 characters:\n", text[:200])

# Getting every unique char in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

st_to_in = {ch: i for i, ch in enumerate(chars)} # String to integer, each char has a number
in_to_st = {i: ch for i, ch in enumerate(chars)} # Integer to string, each number back to its char

encode = lambda s: [st_to_in[c] for c in s] # Turns string into list of numbers using st_to_in
decode = lambda l: "".join([in_to_st[i] for i in l]) # Turns a list of numbers back intoa string using in_to_st

print("Vocabulary size:", vocab_size)
print("Characters:", "".join(chars))

# Test
test_str = "Hello"
encoded = encode(test_str)
print("f\n'{test_str}' encoded:", encoded)
print("Decoded back:", decode(encoded))

data = torch.tensor(encode(text), dtype=torch.long) # Convertig all text into massive list of number through encoder

# Split into training and validation sets, 90% to train on and 10% to check whether model is generalizing or just memorizing
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("Total tokens:", len(data))
print("Train tokens:", len(train_data))
print("Val tokens:", len(val_data))

block_size = 128 # Amount of characters of context the model sees at once to predict the next

def get_batch(split, batch_size=16):
    d = train_data if split == "train" else val_data # Picks the dataset based on whether we are training or validating

    ix = torch.randint(len(d) - block_size, (batch_size,)) # Randomly picks the batch_size starting positions in the data

    x = torch.stack([d[i:i+block_size]for i in ix])  # This is the input batch, where for each starting position i in ix, we grab block_size characters starting there. 

    y = torch.stack([d[i+1:i+block_size+1]for i in ix]) # Same but shifted one, so for every position in the chunk, y tells us exactly "what comes next", giving the model predictions to learn from

    return x, y

# Test
xb, yb = get_batch("train")
print("\nInput shape:", xb.shape)
print("Target shape:", yb.shape)
print("\nFirst training example (input):", decode(xb[0].tolist()))
print("First training example (target):", decode(yb[0].tolist()))

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, block_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim) # Each character id 0 to 64 becomes a learned embed_dim sized vector.

        self.position_embedding = nn.Embedding(block_size, embed_dim) # Each position 0 to block_size-1 becomes a learned embed_dim sized vector, this is how the moel knows where in the seq a character is

        # Added more transformer blocks to get better result, only had one originally
        self.blocks = nn.Sequential(
            TransformerBlock(embed_dim, num_heads, ff_dim),
            TransformerBlock(embed_dim, num_heads, ff_dim),
            TransformerBlock(embed_dim, num_heads, ff_dim),
            TransformerBlock(embed_dim, num_heads, ff_dim),
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size) # Final layer which turns each position's embed_dim sized vector into vocab_size scores, one per possible next character

    def forward(self, idx):
        batch_size, seq_len = idx.shape

        tok_emb = self.token_embedding(idx) # [batch_size, seq_len, embed_dim]
        pos_emb = self.position_embedding(torch.arange(seq_len)) # [seq_len, embed_dim]

        x = tok_emb + pos_emb # Combining both token identity and position information
        x = self.blocks(x) # Run through the transformer block
        logits = self.lm_head(x) # [batch_size, seq_length, vocab_size] - Raw scores per character

        return logits

# Settings for the model
embed_dim = 64
num_heads = 4
ff_dim = 128

model = TinyGPT(vocab_size, embed_dim, num_heads, ff_dim, block_size)

logits = model(xb)  # Re-use xb 

print("Input shape:", xb.shape)
print("Output (logits) shape:", logits.shape)

# Training loop

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for step in range(3000):

    xb, yb = get_batch("train") # Fresh batch of input and target pairs

    logits = model(xb) # Run the batch through the model

    # CrossEntropyLoss expects predictions as [N, num_classes] and targets as [N] but the 3D is (batch, position, vocab_size) and 2D is (batch, position). We have ONE prediction per character POSITION, not just one per example, so we flatten batch and position together into one long list of predictions
    B, T, C = logits.shape # B=batch_size, T="block_size", C="vocab_size"
    logits_flat = logits.view(B*T, C)
    targets_flat = yb.view(B*T)


    loss = nn.functional.cross_entropy(logits_flat, targets_flat) # Compares every prediction position against its true next char target and returns one number which is how wrong the model was o this batch on avg

    optimizer.zero_grad()
    loss.backward() # learning how much to adjust every weight
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

print(f"Final Loss: {loss.item():.4f}")

# Checking validation loss to see if it's much higher than training loss, which would mean the model is memorizing rather than learning
model.eval()
with torch.no_grad():
    xb_val, yb_val = get_batch("val")
    logits = model(xb_val)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    yb_val = yb_val.view(B*T)
    val_loss = nn.functional.cross_entropy(logits, yb_val)

print(f"\nFinal Train Loss: {loss.item():.4f}")
print(f"Validation Loss: {val_loss.item():.4f}") # It isn't!!!

def generate(model, start_text, max_new_tokens=200):
    model.eval()
    idx = torch.tensor([encode(start_text)], dtype=torch.long) # Turns the starting text into token IDS [1, len]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:] # Only keep the last block_size tokens as the model can't see further back then that
        logits = model(idx_cond) # Predictions for every position
        logits = logits[:, -1, :] # Only prediction for the NEXT character

        probs = torch.softmax(logits, dim=-1) # Turn scores into probabilities
        next_id = torch.multinomial(probs, num_samples=1) # Randomly sample one character, weighed by those probs

        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist())

generated = generate(model, start_text="ROMEO:", max_new_tokens=300)
print("\n--- Generated Text ---")
print(generated)