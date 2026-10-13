"""
Bigram character-level language model, trained on tiny Shakespeare.

Day 3 of my AI infra learning journey — blind rewrite of nanoGPT's bigram baseline.
Only looks at the immediately preceding character to predict the next.

Usage:
    python bigram.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- Hyperparameters ---------------------
BLOCK_SIZE = 8  # context length (how many chars per sample)
BATCH_SIZE = 32  # how many sequences per training step
NUM_STEPS = 3000
LEARNING_RATE = 1e-3
EVAL_INTERVAL = 300
MAX_NEW_TOKENS = 500
TORCH_SEED = 1337


# --------------------- Data pipeline ---------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    """string -> list[int]"""
    return [stoi[c] for c in s]


def decode(ids):
    """list[int] -> string"""
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    """Sample a batch of (B, T) input/target pairs from the data."""
    source = train_data if split == "train" else val_data
    starts = torch.randint(0, len(source) - block_size, (batch_size,))
    xb = torch.stack([source[i : i + block_size] for i in starts])
    yb = torch.stack([source[i + 1 : i + block_size + 1] for i in starts])
    return xb, yb


# --------------------- Model ---------------------
class BigramLM(nn.Module):
    """Pure bigram model: each token's embedding IS the logits for the next token."""

    def __init__(self, vocab_size):
        super().__init__()
        # The embedding table is (V, V): row i = logits for the char after char i.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) LongTensor of token ids
        targets: (B, T) LongTensor of next-token ids, or None (for inference)

        Returns:
            logits: (B, T, V)
            loss:   scalar FloatTensor or None
        """
        logits = self.token_embedding_table(idx)  # (B, T, V)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            # cross_entropy expects (N, C) and (N,); flatten the batch/time dims
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        idx: (B, T) current context
        Appends max_new_tokens by autoregressive sampling.
        Returns (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # (B, T, V)
            logits = logits[:, -1, :]  # take last timestep: (B, V)
            probs = F.softmax(logits, dim=-1)  # (B, V)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)  # (B, T+1)
        return idx


# --------------------- Training ---------------------
def train(model, num_steps=NUM_STEPS, lr=LEARNING_RATE):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(num_steps):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % EVAL_INTERVAL == 0 or step == num_steps - 1:
            print(f"step {step:5d} | train loss {loss.item():.4f}")
    return loss.item()


# --------------------- Entry point ---------------------
if __name__ == "__main__":
    torch.manual_seed(TORCH_SEED)

    print(f"vocab_size = {vocab_size}")
    print(
        f"data: {len(data):,} tokens (train {len(train_data):,}, val {len(val_data):,})"
    )
    print(f"theoretical initial loss ≈ ln({vocab_size}) = {math.log(vocab_size):.4f}\n")

    model = BigramLM(vocab_size)
    final_loss = train(model)
    print(f"\nFinal train loss: {final_loss:.4f}")

    print(f"\n--- Generated sample ({MAX_NEW_TOKENS} chars) ---")
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=MAX_NEW_TOKENS)
    print(decode(out[0].tolist()))
