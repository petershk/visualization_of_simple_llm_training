import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)

vocab = {
    "the": 0,
    "cat": 1,
    "chased": 2,
    "mouse": 3,
    "dog": 4,
    "saw": 5
}

#Each token withh have 4 features for easy inspection
embedding = nn.Embedding(
    num_embeddings=len(vocab), 
    embedding_dim=4
    )

#Convert our sentence into token IDs
words = ["the", "cat", "chased", "the", "mouse"]
token_ids = torch.tensor([vocab[word] for word in words])

#Look up the embeddings
X = embedding(token_ids)

print ("Token IDS:", token_ids)
print ("X SHape:" , X.shape)

for word, vector in zip(words, X):
    print(f"{word}: {vector.detach().tolist()}")

#Three separate learniable transformations
W_Q = nn.Linear(4,4, bias = False)
W_K = nn.Linear(4,4, bias = False)
W_V = nn.Linear(4,4, bias = False)

Q = W_Q(X)
K = W_K(X)
V = W_V(X)

print("\nQ Shape:", Q.shape)
print("K Shape:", K.shape)
print("V Shape:", V.shape)

print("\nQuery for cat:")
print(Q[1].detach())

print("\nValue for cat:")
print(V[1].detach())

print("\nKey for cat:")
print(K[1].detach())

#Calculate Attention
scores = Q @ K.T / math.sqrt(4)
weights = F.softmax(scores, dim=-1)
output = weights @ V

print("\nAttention Weights for 'cat':")
print(weights[1].detach())
print("\nAttention Output for 'cat':")
print(output[1].detach())