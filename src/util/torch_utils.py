from torch import nn


class TorchStandardScaler:
    """
    Used to compute mean and standard deviation of a dataset along axis 0 in order to transform tensors to mean=0
    and sigma=1 based on a training set.
    """

    def __init__(self, x, keepdim=True):
        self.mean = x.mean(0, keepdim=keepdim)
        self.std = x.std(0, unbiased=False, keepdim=keepdim) + 1e-8

    def transform(self, x):
        x -= self.mean
        x /= self.std
        return x

    def inverse_transform(self, x):
        x *= self.std
        x += self.mean
        return x


def init_weights(m):
    """
    Used to init the weights of a linear layer using Xavier uniform initialization. Can be applied to a full nn.Module
    by applying it, e.g., model.apply(init_weights).

    :param m: The linear layer to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def enable_dropout(model):
    """
    Function to only set Dropout layers to train in a given module.

    Source: https://stackoverflow.com/a/63397197
    """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
