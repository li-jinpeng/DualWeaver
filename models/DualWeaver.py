import random
import torch
from torch import nn
from torch.nn.parameter import Parameter

from models.FeatureWeaver import (
    WeaverCNN,
    WeaverMLP,
)


class Model(nn.Module):
    def __init__(self, configs, ltm):
        super().__init__()
        self.configs = configs
        self.ltm = ltm

        FeatureWeaver = {
            "WeaverMLP": WeaverMLP.FeatureWeaver,
            "WeaverCNN": WeaverCNN.FeatureWeaver,
        }[self.configs.adapter]
        self.feature_weaver = FeatureWeaver(configs)

        self.a = Parameter(torch.ones(configs.input_channel))
        self.b = Parameter(torch.ones(configs.input_channel))

    def forward(self, batch_x, batch_y=None):
        B = batch_x.shape[0]
        M = batch_x.shape[-1]

        x0 = batch_x.permute(0, 2, 1)  # B C L
        x0 = x0.reshape(-1, x0.shape[-1])

        x1 = self.a * batch_x + self.feature_weaver(batch_x)
        x2 = -self.b * batch_x + self.feature_weaver(batch_x)
        batch_x = torch.cat([x1, x2], dim=0)
        batch_x = batch_x.permute(0, 2, 1)  # 2B C L
        batch_x = batch_x.reshape(-1, batch_x.shape[-1])

        if self.training:
            y0 = batch_y.permute(0, 2, 1)  # B C L
            y0 = y0.reshape(-1, y0.shape[-1])
            y1 = self.a * batch_y + self.feature_weaver(batch_y)
            y2 = -self.b * batch_y + self.feature_weaver(batch_y)
            batch_y = torch.cat([y1, y2], dim=0)
            batch_y = batch_y.permute(0, 2, 1)  # 2B C L
            batch_y = batch_y.reshape(-1, batch_y.shape[-1])

        if "timer" in self.configs.model:
            if self.training:
                outputs = self.ltm(
                    input_ids=batch_x,
                    labels=batch_y,
                )
                losses = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss1, loss2 = torch.chunk(losses, 2, dim=0) # B*C P
                loss1 = loss1.mean(dim=-1).reshape(B, M)
                loss2 = loss2.mean(dim=-1).reshape(B, M)

                with torch.no_grad():
                    self.ltm.eval()
                    outputs_origin = self.ltm(
                        input_ids=x0,
                        labels=y0,
                    )
                    loss_origin = (
                        outputs_origin["loss"]
                        if isinstance(outputs_origin, dict)
                        else outputs_origin[0]
                    )
                    loss_origin = loss_origin.mean(dim=-1).reshape(B, M)
                    self.ltm.train()

                re_loss = 2 * (loss1 + loss2) / (self.a.unsqueeze(0) + self.b.unsqueeze(0)) ** 2
                re_flag = re_loss < loss_origin
                return (loss1.mean() + loss2.mean()) / 2 + torch.max(re_loss, loss_origin.detach()).mean(), re_flag
            else:
                outputs = self.ltm.generate(
                    batch_x,
                    max_new_tokens=self.configs.test_pred_len,
                )
                predictions = outputs.reshape(2 * B, -1, outputs.shape[-1])
                predictions = predictions.permute(0, 2, 1)
                y1, y2 = torch.chunk(predictions, 2, dim=0)
                predictions = (y1 - y2) / (self.a + self.b)
                return predictions
        elif "sundial" in self.configs.model:
            if self.training:
                outputs = self.ltm(
                    input_ids=batch_x,
                    labels=batch_y,
                    pred_len=self.configs.test_pred_len,
                )
                losses = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss1, loss2 = torch.chunk(losses, 2, dim=0) # B*C P
                loss1 = loss1.reshape(B*M, -1).mean(dim=-1).reshape(B, M)
                loss2 = loss2.reshape(B*M, -1).mean(dim=-1).reshape(B, M)

                with torch.no_grad():
                    self.ltm.eval()
                    outputs_origin = self.ltm(
                        input_ids=x0,
                        labels=y0,
                    )
                    loss_origin = (
                        outputs_origin["loss"]
                        if isinstance(outputs_origin, dict)
                        else outputs_origin[0]
                    )
                    loss_origin = loss_origin.reshape(B*M, -1).mean(dim=-1).reshape(B, M)
                    self.ltm.train()

                re_loss = 2 * (loss1 + loss2) / (self.a.unsqueeze(0) + self.b.unsqueeze(0)) ** 2
                re_flag = re_loss < loss_origin
                return (loss1.mean() + loss2.mean()) / 2 + torch.max(re_loss, loss_origin.detach()).mean(), re_flag
            else:
                outputs = self.ltm.generate(
                    batch_x,
                    max_new_tokens=self.configs.test_pred_len,
                    num_samples=self.configs.test_n_sample,
                )
                predictions = outputs.reshape(
                    2 * B, -1, outputs.shape[1], outputs.shape[-1]
                )
                predictions = predictions.permute(0, 2, 1, 3)  # B N C L
                predictions = predictions.permute(0, 1, 3, 2)  # B N L C
                y1, y2 = torch.chunk(predictions, 2, dim=0)
                predictions = (y1 - y2) / (self.a + self.b)
                predictions = predictions.permute(1, 0, 2, 3)
                predictions = predictions.permute(1, 0, 2, 3)
                predictions = predictions.mean(dim=1)
                return predictions
        else:
            raise NotImplementedError