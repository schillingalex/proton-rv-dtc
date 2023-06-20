import random
import numpy as np


def make_deterministic(seed: int = 0, include_torch: bool = True):
    """
    Call to make all native Python, numpy, and optionally PyTorch operations deterministic for a given random seed.

    Only if PyTorch is included will it be imported dynamically to not overload implementations without PyTorch with
    this large library.

    :param seed: Random seed to use. Optional, default: 0
    :param include_torch: Should PyTorch be included? Optional, default: True
    """
    random.seed(seed)
    np.random.seed(seed)

    if include_torch:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
