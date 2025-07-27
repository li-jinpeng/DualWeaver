import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs, ltm):
        super().__init__()
        self.configs = configs
        self.encoder = nn.Linear(configs.input_channel, configs.output_channel)
        self.decoder = nn.Linear(configs.output_channel, configs.input_channel)
        self.criterion = nn.MSELoss()
        self.ltm = ltm

    def forward(
        self, batch_x, batch_y=None, loss_masks=None, mask_y=None, training=False
    ):
        B = batch_x.shape[0]

        means = batch_x.mean(1, keepdim=True).detach()
        stdev = batch_x.std(dim=1, keepdim=True, unbiased=False).detach()
        stdev = torch.where(
            stdev > 1e-2, stdev, torch.tensor(1e-2, device=batch_x.device)
        )
        batch_x = (batch_x - means) / stdev
        if training:
            batch_y = (batch_y - means) / stdev

        batch_x = self.encoder(batch_x)
        batch_x = batch_x.permute(0, 2, 1)
        batch_x = batch_x.reshape(-1, batch_x.shape[-1])

        if "timer" in self.configs.model:
            outputs = self.ltm.generate(
                batch_x,
                max_new_tokens=self.configs.test_pred_len,
            )
            predictions = outputs.reshape(B, -1, outputs.shape[-1])
            predictions = predictions.permute(0, 2, 1)
            predictions = self.decoder(predictions)
            if training:
                pred_begin = self.configs.seq_len - self.configs.input_token_len
                batch_y = batch_y[
                    :, pred_begin : pred_begin + self.configs.output_token_len, :
                ]
                predictions = predictions[:, : self.configs.output_token_len, :]
                loss = self.criterion(batch_y, predictions)
                return loss
            predictions = predictions * stdev + means
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
            if training:
                pred_begin = self.configs.seq_len - self.configs.input_token_len
                batch_y = batch_y[
                    :, pred_begin : pred_begin + self.configs.test_pred_len, :
                ]
                predictions = predictions.mean(dim=1)  # B L C
                predictions = predictions[:, : self.configs.test_pred_len, :]
                loss = self.criterion(batch_y, predictions)
                return loss
            predictions = predictions.permute(1, 0, 2, 3)
            predictions = predictions * stdev + means
            predictions = predictions.permute(1, 0, 2, 3)
            predictions = predictions.mean(dim=1)
            return predictions
        else:
            raise NotImplementedError
