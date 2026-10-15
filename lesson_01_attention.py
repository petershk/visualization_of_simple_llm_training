import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)

#We have three tokens:
tokens = ["cat", "chased", "mouse"]

Q = torch.tensor([
    [0.0,2.0], #cat
    [0.0,1.0], #chased
    [1.0,1.0], #mouse
])

K = torch.tensor([
    [1.0, 0.0],  # cat
    [0.0, 1.0],  # chased
    [1.0, 1.0]   # mouse
])

V = torch.tensor([
    [20.0,0.0],
    [0.0,40.0],
    [60.0,60.0],
])

print("=== INPUT ===")
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

#STEP 1: Cmompare all queries against all keys
scores = Q @ K.T

print("\n=== RAW SCORES ===")
print(scores)   

#STEP 2: Scale the dot products
d_k = K.shape[-1]
scores = scores /math.sqrt(d_k)

print("\n=== SCALED SCORES ===")
print(scores)

#STEP 3: Apply the softmax to get the attention weights
weights = torch.softmax(scores, dim=-1)

print("\n=== ATTENTION WEIGHTS ===")
print(weights)

#STEP 4: Combine value vectors
output = weights @ V

print("\n=== ATTENTION OUTPUT ===")
print(output)

#Display results for each token
for i, token in enumerate(tokens):
    print(f"\nToken: {token}")
    for j, other in enumerate(tokens):
        print(
            f"  Attention to {other}: "
            f"{weights[i, j].item():.1%}"
        )
    print(f"  Output vector: {output[i].tolist()}") 