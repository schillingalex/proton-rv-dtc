import torch
from torch import nn


class RegressionNet(nn.Module):
    """
    Fully connected network with 3 hidden layers, variable activation and variable number of output neurons
    for separate regression tasks. Additionally predicts aleatoric uncertainty (variance).
    """

    def __init__(self, feature_count: int, outputs: int = 2, dropout: float = 0.05, activation: str = "sigmoid"):
        super(RegressionNet, self).__init__()
        activation_layer = nn.Sigmoid()
        if activation == "tanh":
            activation_layer = nn.Tanh()
        elif activation == "relu":
            activation_layer = nn.ReLU()

        self.h1 = nn.Sequential(nn.Linear(feature_count, 1024), activation_layer, nn.Dropout(dropout))
        self.h2 = nn.Sequential(nn.Linear(1024, 512), activation_layer, nn.Dropout(dropout))
        self.h3 = nn.Sequential(nn.Linear(512, 128), activation_layer, nn.Dropout(dropout))

        self.regressor = nn.Linear(128, outputs)
        self.log_var_predictor = nn.Linear(128, outputs)

    def forward(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        return self.regressor(x), torch.exp(self.log_var_predictor(x))
