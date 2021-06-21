import numpy as np
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )


    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        return self.fc(x)
