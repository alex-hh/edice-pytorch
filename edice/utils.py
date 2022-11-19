import numpy as np
import torch


def isnumeric(v):
    """Required because we support non numeric metrics for wandb."""
    return isinstance(v, float) or isinstance(v, int) or np.isscalar(v)


def get_device(force_cpu=False):
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device", device, flush=True)
    return device


def set_seeds(seed):
    if seed is not None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        # also see pytorch lightning seed everything
        print(f"seeding {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
