import torch


def torch_load(path, *args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return torch.load(path, *args, **kwargs)
