import torch

def norm(A):
    B = torch.zeros_like(A)
    if sum(A):
        for idx, each in enumerate(A):
            B[idx] = each / sum(A)
    return B
