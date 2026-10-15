"""Lesson 7: causal multi-head attention with a verified KV cache.

Run: uv run python lesson_07_kv_cache.py
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

SEED = 123
EMBED_DIM = 32
NUM_HEADS = 4
MAX_LEN = 16
TRAINING_STEPS = 500
LEARNING_RATE = 0.003

SENTENCES = [
    "the cat chased the mouse",
    "the dog chased the ball",
    "the cat ate the fish",
    "the dog ate the food",
    "the fish ate the blargh",
]

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
words = sorted({word for sentence in SENTENCES for word in sentence.split()} | {"<BOS>", "<EOS>"})
stoi = {word: i for i, word in enumerate(words)}
itos = {i: word for word, i in stoi.items()}
encoded = [[stoi[t] for t in ["<BOS>", *sentence.split(), "<EOS>"]] for sentence in SENTENCES]
assert len({len(seq) for seq in encoded}) == 1, "Variable-length sentences need padding or separate batches"
X = torch.tensor([seq[:-1] for seq in encoded], device=DEVICE)
Y = torch.tensor([seq[1:] for seq in encoded], device=DEVICE)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads:
            raise ValueError("Embedding dimension must be divisible by number of heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim)
        self.last_attention = None

    def forward(self, x, cache=None):
        B, T, C = x.shape

        def split_heads(t):
            return t.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.query(x))
        k = split_heads(self.key(x))
        v = split_heads(self.value(x))

        past_len = 0
        if cache is not None:
            past_k, past_v = cache
            past_len = past_k.shape[2]
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        query_positions = past_len + torch.arange(T, device=x.device)
        key_positions = torch.arange(k.shape[2], device=x.device)
        future_mask = key_positions[None, :] > query_positions[:, None]
        scores = scores.masked_fill(future_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        self.last_attention = weights.detach()

        out = weights @ v
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.output(out), (k, v)


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, dim=32, num_heads=4, max_len=16):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_len, dim)
        self.attention = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, tokens, cache=None, use_cache=False):
        B, T = tokens.shape
        past_len = 0 if cache is None else cache[0].shape[2]
        if past_len + T > self.position_embedding.num_embeddings:
            raise ValueError("Input exceeds positional context; this lesson does not use sliding-window caching")
        positions = torch.arange(past_len, past_len + T, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        attn_out, new_cache = self.attention(self.norm1(x), cache=cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        logits = self.output(x)
        return (logits, new_cache) if use_cache else logits


model = MiniGPT(len(words), EMBED_DIM, NUM_HEADS, MAX_LEN).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def train():
    model.train()
    for step in range(TRAINING_STEPS + 1):
        logits = model(X)
        loss = F.cross_entropy(logits.reshape(-1, len(words)), Y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.4f}")
    model.eval()


def show_predictions():
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(X)[:, -2, :], dim=-1)
        for i, sentence in enumerate(SENTENCES):
            pred = itos[probs[i].argmax().item()]
            correct = itos[Y[i, -2].item()]
            print(f"{' '.join(sentence.split()[:-1]):25s} predicted={pred:8s} "
                  f"correct={correct:8s} p(correct)={probs[i, Y[i,-2]].item():.1%}")


def inspect_heads(sentence_index=1):
    model.eval()
    with torch.no_grad():
        model(X)
    attention = model.attention.last_attention
    tokens = ["<BOS>", *SENTENCES[sentence_index].split()]
    query_index = len(tokens) - 2  # Last 'the' predicts the final ordinary word
    print(f"\nHEAD ATTENTION: {SENTENCES[sentence_index]}")
    print("Shape:", tuple(attention.shape))
    for head in range(NUM_HEADS):
        print(f"Head {head + 1}:")
        for token, weight in zip(tokens[:query_index+1], attention[sentence_index, head, query_index, :query_index+1]):
            print(f"  {token:8s} {weight.item():6.1%}")


def generate_cached(prompt, max_new_tokens=10):
    """Prefill once, decode one token at a time, and verify against full recomputation."""
    model.eval()
    prompt_words = prompt.split()
    unknown = [w for w in prompt_words if w not in stoi]
    if unknown:
        raise ValueError(f"Unknown tokens: {unknown}")
    tokens = torch.tensor([[stoi["<BOS>"], *(stoi[w] for w in prompt_words)]], device=DEVICE)
    cache = None
    print(f"\nPrompt: {prompt}")

    with torch.no_grad():
        for step in range(max_new_tokens):
            if tokens.shape[1] > MAX_LEN:
                print("Context limit reached")
                break

            if cache is None:
                cached_logits, cache = model(tokens, use_cache=True)
                mode = "PREFILL"
            else:
                cached_logits, cache = model(tokens[:, -1:], cache=cache, use_cache=True)
                mode = "DECODE"

            # Correctness verification only. Remove this expensive reference
            # calculation when benchmarking a cached generator.
            full_logits = model(tokens)
            cached_next = cached_logits[:, -1, :]
            full_next = full_logits[:, -1, :]
            diff = (cached_next - full_next).abs().max().item()
            torch.testing.assert_close(cached_next, full_next, rtol=1e-4, atol=1e-4)
            next_id = cached_next.argmax(dim=-1, keepdim=True)
            word = itos[next_id.item()]
            print(f"Step {step + 1}: {mode:7s} | word={word:8s} | "
                  f"cache={cache[0].shape[2]} tokens | max logit diff={diff:.8f}")

            if word == "<EOS>":
                break
            tokens = torch.cat((tokens, next_id), dim=1)

    return " ".join(itos[i] for i in tokens[0].tolist() if itos[i] not in {"<BOS>", "<EOS>"})


if __name__ == "__main__":
    print("Vocabulary:", stoi)
    print("Device:", DEVICE)
    print("Heads:", NUM_HEADS)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Training tensors:", tuple(X.shape), tuple(Y.shape))
    print("\nBEFORE TRAINING:")
    show_predictions()
    print("\nTRAINING:")
    train()
    print("\nAFTER TRAINING:")
    show_predictions()
    inspect_heads()
    print("\nKV CACHE EXPERIMENT:")
    for p in ("the cat", "the dog", "the fish"):
        print("Result:", generate_cached(p))
