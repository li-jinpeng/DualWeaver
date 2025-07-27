import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs, ltm):
        super().__init__()
        self.configs = configs
        self.ltm = ltm

    def forward(
        self, batch_x, batch_y=None, loss_masks=None, mask_y=None, training=False
    ):
        if "timer" in self.configs.model:
            if training:
                outputs = self.ltm(
                    input_ids=batch_x,
                    labels=batch_y,
                    loss_masks=loss_masks,
                    no_grad=False,
                )
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                return loss.mean()
            else:
                predictions = self.ltm.generate(
                    batch_x,
                    max_new_tokens=self.configs.test_pred_len,
                )
                return predictions
        elif "sundial" in self.configs.model:
            if training:
                outputs = self.ltm(
                    input_ids=batch_x,
                    labels=batch_y,
                    mask_y=mask_y,
                    loss_masks=loss_masks,
                    no_grad=False,
                )
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                return loss.mean()
            else:
                predictions = self.ltm.generate(
                    batch_x,
                    max_new_tokens=self.configs.test_pred_len,
                    num_samples=self.configs.test_n_sample,
                )
                predictions = predictions.mean(dim=1)
                return predictions
        else:
            raise NotImplementedError
