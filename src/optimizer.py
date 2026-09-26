import torch

def make_optimizer(type,lr,params):
    if type == "SGD":
        return torch.optim.SGD(
            params=params,
            lr=lr
        )
    elif type == "AdamW":
        return torch.optim.AdamW(
            params=params,
            lr=lr,
        )
    