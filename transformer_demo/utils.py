import torch


def subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0)
