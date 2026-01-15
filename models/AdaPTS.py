import math
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs, ltm):
        super().__init__()
        self.configs = configs
        output_channel = min(max(2 ** (int(math.log2(configs.input_channel)) + 1), 32), 512)
        self.output_channel = output_channel

        self.encoder = nn.Linear(configs.input_channel, output_channel)
        self.decoder = nn.Linear(output_channel, configs.input_channel)
        self.criterion = nn.MSELoss()
        self.ltm = ltm

    def forward(
        self, batch_x, batch_y=None
    ):
        B, L, C = batch_x.shape

        batch_x = self.encoder(batch_x)
        batch_x = batch_x.permute(0, 2, 1)
        batch_x = batch_x.reshape(-1, batch_x.shape[-1]) # B*C L

        if "timer" in self.configs.model:
            outputs = self.ltm.generate(
                batch_x,
                max_new_tokens=self.configs.test_pred_len,
            )
            predictions = outputs.reshape(B, -1, outputs.shape[-1])
            predictions = predictions.permute(0, 2, 1)
            predictions = self.decoder(predictions)
            if self.training:
                pred_begin = self.configs.seq_len - self.configs.input_token_len
                batch_y = batch_y[
                    :, pred_begin : pred_begin + self.configs.output_token_len, :
                ]
                predictions = predictions[:, : self.configs.output_token_len, :]
                loss = self.criterion(batch_y, predictions)
                return loss
            return predictions
        elif "sundial" in self.configs.model:
            outputs = self.ltm.generate(
                batch_x,
                max_new_tokens=self.configs.test_pred_len,
                num_samples=self.configs.test_n_sample,
            )
            predictions = outputs.reshape(B, -1, outputs.shape[1], outputs.shape[-1])
            predictions = predictions.permute(0, 2, 1, 3)  # B N C L
            predictions = predictions.permute(0, 1, 3, 2)  # B N L C
            predictions = self.decoder(predictions)
            if self.training:
                pred_begin = self.configs.seq_len - self.configs.input_token_len
                batch_y = batch_y[
                    :, pred_begin : pred_begin + self.configs.test_pred_len, :
                ]
                predictions = predictions.mean(dim=1)  # B L C
                predictions = predictions[:, : self.configs.test_pred_len, :]
                loss = self.criterion(batch_y, predictions)
                return loss
            predictions = predictions.permute(1, 0, 2, 3)
            predictions = predictions.permute(1, 0, 2, 3)
            predictions = predictions.mean(dim=1)
            return predictions
        else:
            raise NotImplementedError
