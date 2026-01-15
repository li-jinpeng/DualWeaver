import math
import torch
from torch import nn
from torch.nn.parameter import Parameter


class FeatureWeaver(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        output_channel = min(
            max(2 ** (int(math.log2(configs.input_channel)) + 1), 32), 512
        )

        self.fc1 = nn.Linear(configs.input_channel, output_channel)
        self.fc2 = nn.Linear(output_channel, configs.input_channel)
        self.silu = nn.SiLU()

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = torch.dropout(self.silu(self.fc1(x)), p=0.1, train=self.training)
        x = self.fc2(x)
        return x
