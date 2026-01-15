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

        self.conv1 = nn.Conv1d(
            configs.input_channel,
            output_channel,
            5,
            1,
            2,
            1,
            padding_mode="replicate",
        )
        self.layernorm1 = nn.LayerNorm(output_channel)

        self.conv2 = nn.Conv1d(
            output_channel,
            configs.input_channel,
            5,
            1,
            2,
            1,
            padding_mode="replicate",
        )
        self.layernorm_2 = nn.LayerNorm(configs.input_channel)

        self.silu = nn.SiLU()
        self.dropout = nn.Dropout1d(p=0.1)

        self.fc = nn.Linear(configs.input_channel, configs.input_channel)

        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B C L

        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # B L C
        x = self.layernorm1(x)
        x = self.silu(x)
        if self.training:
            x = self.dropout(x)

        x = x.permute(0, 2, 1)  # B C L
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # B L C
        x = self.layernorm_2(x)
        x = self.silu(x)
        if self.training:
            x = self.dropout(x)

        x = self.fc(x)
        return x

