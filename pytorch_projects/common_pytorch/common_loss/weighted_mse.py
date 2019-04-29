import torch

def weighted_mse_loss(input, target, weights, size_average):
    out = (input - target) ** 2
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()

def weighted_l1_loss(input, target, weights, size_average):
    # import ipdb; ipdb.set_trace
    out = torch.abs(input - target)
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()
