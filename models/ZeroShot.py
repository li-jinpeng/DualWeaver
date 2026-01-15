import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs, ltm):
        super().__init__()
        self.configs = configs
        self.ltm = ltm

    def forward(self, batch_x):
        if "timer" in self.configs.model:
            predictions = self.ltm.generate(
                batch_x,
                max_new_tokens=self.configs.test_pred_len,
            )
            return predictions
        elif "sundial" in self.configs.model:
            predictions = self.ltm.generate(
                batch_x,
                max_new_tokens=self.configs.test_pred_len,
                num_samples=self.configs.test_n_sample,
            )
            predictions = predictions.mean(dim=1)
            return predictions
        else:
            raise NotImplementedError
