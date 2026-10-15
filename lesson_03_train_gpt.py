
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(123)

# --------------------------------------
# 1. TRAINING DATA
# --------------------------------------

sentences = [
    "the cat chased the mouse",
    "the dog chased the ball",
    "the cat ate the fish",
    "the dog ate the food",
    "the fish ate the blargh",
]

# Build our vocabulary
words = sorted(set(
    word
    for sentence in sentences
    for word in sentence.split())
    | {"<BOS>", "<EOS>"}
)

stoi = {word: i for i, word in enumerate(words)}
itos = {i: word for word, i in stoi.items()}

vocab_size = len(words)

print("Vocabulary:", stoi)

# Input = first four words
# Target = fifth word
X = []
Y = []

for sentence in sentences:
    tokens = ["<BOS>"] + sentence.split() + ["<EOS>"]
    ids = [stoi[word] for word in tokens]

    X.append(ids[:-1])
    Y.append(ids[1:])

X = torch.tensor(X)
Y = torch.tensor(Y)

# --------------------------------------
# 2. CAUSAL SELF-ATTENTION
# --------------------------------------

class CausalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(Q.shape[-1])

        # Prevent attending to future tokens
        seq_len = x.shape[1]

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        self.last_attention = weights.detach()

        return weights @ V


# --------------------------------------
# 3. MINI LANGUAGE MODEL
# --------------------------------------

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, dim=32, max_len=16):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_len, dim)

        self.attention = CausalAttention(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.output = nn.Linear(dim, vocab_size)

    def forward(self, tokens):
        B, T = tokens.shape

        positions = torch.arange(T, device=tokens.device)

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
        )

        # Residual attention block
        x = x + self.attention(self.norm1(x))

        # Residual feed-forward block
        x = x + self.mlp(self.norm2(x))

        # Predict next word using the last position
        logits = self.output(x)

        return logits


# --------------------------------------
# 4. INITIALIZE MODEL
# --------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MiniGPT(vocab_size).to(device)

X = X.to(device)
Y = Y.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.003
)

print("Device:", device)
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# --------------------------------------
# 5. INSPECT PREDICTIONS
# --------------------------------------


def show_predictions():
    model.eval()

    with torch.no_grad():
        logits = model(X)

        # Only inspect the final-position predictions
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)

        for i, sentence in enumerate(sentences):
            predicted_id = probs[i].argmax().item()
            correct_id = Y[i, -1].item()

            probability = probs[i, correct_id].item()

            print(
                f"{' '.join(sentence.split()[:-1]):25s}"
                f" predicted={itos[predicted_id]:6s}"
                f" correct={itos[correct_id]:6s}"
                f" confidence={probability:.1%}"
            )

    model.train()


print("\nBEFORE TRAINING:")
show_predictions()


# --------------------------------------
# 6. TRAIN
# --------------------------------------

model.train()

for step in range(501):
    logits = model(X)

    B, T, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        Y.reshape(B * T)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.4f}")
        show_predictions()


# --------------------------------------
# 7. TEST AFTER TRAINING
# --------------------------------------

print("\nAFTER TRAINING:")
model.eval()

with torch.no_grad():
    model(X)

attention = model.attention.last_attention

print("\nATTENTION FOR FINAL WORD POSITION:")

for i, sentence in enumerate(sentences):
    tokens = ["<BOS>"] + sentence.split()
    
    print(f"\n{sentence}")

    for token, weight in zip(tokens, attention[i, -1]):
        print(f"    {token:8s} {weight.item():.1f}")



def predict_next(prompt):
    model.eval()

    word_list = prompt.split()

    token_ids = torch.tensor(
        [[stoi[word] for word in word_list]],
        device=device
    )

    with torch.no_grad():
        logits = model(token_ids)

        # Select the last position's prediction
        last_logits = logits[:, -1, :]

        probs = F.softmax(last_logits, dim=-1)

    top_probs, top_ids = torch.topk(probs[0], k=3)

    print(f"\nPrompt: {prompt}")

    for probability, token_id in zip(top_probs, top_ids):
        print(
            f"  {itos[token_id.item()]:8s}: "
            f"{probability.item():.1%}"
        )



predict_next("the mouse chased the")
predict_next("the fish ate the")
predict_next("the ball chased the")
predict_next("the dog ate the")

show_predictions()


print("\nALL POSITION PREDICTIONS:")

model.eval()

with torch.no_grad():
    logits = model(X)
    predictions = logits.argmax(dim=-1)

for i, sentence in enumerate(sentences):
    print(f"\n{sentence}")

    input_words = sentence.split()[:-1]

    for j, word in enumerate(input_words):
        predicted = itos[predictions[i, j].item()]
        expected = itos[Y[i, j].item()]

        print(
            f"  After '{word:8s}' "
            f"predict={predicted:8s} "
            f"expected={expected:8s}"
        )

def generate(prompt, max_new_tokens = 20):
    model.eval()

    ids = [stoi["<BOS>"]]
    ids += [stoi[word] for word in prompt.split()]

    tokens = torch.tensor([ids], device=device)

    max_context = model.position_embedding.num_embeddings

    with torch.no_grad():
        for step in range(max_new_tokens):

            #Keep only the tokens that fit the context
            context = tokens[:, -max_context:]

            logits = model(context)

            next_logits = logits[:, -1, :]

            #pick the most likely token
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            word = itos[next_id.item()]

            print(f"Step {step+1}: generated '{word}'")

            if(word == "<EOS>"):
                break

            #Append newly generated token to input
            tokens = torch.cat([tokens, next_id], dim=1)

    result = [
        itos[i] for i in tokens[0].tolist()
        if itos[i] not in ("<BOS>", "<EOS>")
    ]

    return " ".join(result)

print("\nGENERATED TEXT:")
print(generate("\nthe cat", max_new_tokens=10))

print(generate("\nthe dog", max_new_tokens=10))

print(generate("\nthe fish", max_new_tokens=10))
