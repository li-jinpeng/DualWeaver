import torch
from torch import nn
from torch.nn.parameter import Parameter


class FeatureWeaver(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.fc1 = nn.Linear(configs.input_channel, configs.output_channel)
        self.fc2 = nn.Linear(configs.output_channel, configs.input_channel)
        self.silu = nn.SiLU()

    def forward(self, x, training=False):
        x = torch.dropout(self.silu(self.fc1(x)), p=0.1, train=bool(training))
        x = self.fc2(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, ltm):
        super().__init__()
        self.configs = configs
        self.ltm = ltm
        self.feature_weaver = FeatureWeaver(configs)
        self.a = Parameter(torch.ones(1, configs.input_channel))
        self.b = Parameter(torch.ones(1, configs.input_channel))
        self.criterion = nn.MSELoss()

    def forward(
        self, batch_x, batch_y=None, loss_masks=None, mask_y=None, training=False
    ):
        means = batch_x.mean(1, keepdim=True).detach()
        stdev = batch_x.std(dim=1, keepdim=True, unbiased=False).detach()
        stdev = torch.where(
            stdev > 1e-2, stdev, torch.tensor(1e-2, device=batch_x.device)
        )

        batch_x = (batch_x - means) / stdev
        x0 = batch_x.permute(0, 2, 1)  # B C L
        x0 = x0.reshape(-1, x0.shape[-1])

        x1 = self.a * batch_x + self.feature_weaver(batch_x, training)
        x2 = -self.b * batch_x + self.feature_weaver(batch_x, training)
        batch_x = torch.cat([x1, x2], dim=0)
        B = batch_x.shape[0]
        M = batch_x.shape[-1]
        batch_x = batch_x.permute(0, 2, 1)  # B C L
        batch_x = batch_x.reshape(-1, batch_x.shape[-1])
        batch_x = batch_x

        if training:
            batch_y = (batch_y - means) / stdev
            y0 = batch_y.permute(0, 2, 1)  # B C L
            y0 = y0.reshape(-1, y0.shape[-1])
            y1 = self.a * batch_y + self.feature_weaver(batch_y, training)
            y2 = -self.b * batch_y + self.feature_weaver(batch_y, training)
            batch_y = torch.cat([y1, y2], dim=0)
            batch_y = batch_y.permute(0, 2, 1)
            batch_y = batch_y.reshape(-1, batch_y.shape[-1])

        if "timer" in self.configs.model:
            if training:
                outputs = self.ltm(
                    input_ids=batch_x,
                    labels=batch_y,
                )
                losses = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss1, loss2 = torch.chunk(losses, 2, dim=0)
                loss1 = loss1.reshape(B, M, -1).permute(0, 2, 1).reshape(-1, M)
                loss2 = loss2.reshape(B, M, -1).permute(0, 2, 1).reshape(-1, M)

                with torch.no_grad():
                    outputs_origin = self.ltm(
                        input_ids=x0,
                        labels=y0,
                    )
                    loss_origin = outputs_origin["loss"] if isinstance(outputs, dict) else outputs[0]

                return loss1.mean() + loss2.mean() + torch.max((2 * (loss1 + loss2) / (self.a + self.b) ** 2).mean(), loss_origin.mean().detach())
            else:
                outputs = self.ltm.generate(
                    batch_x,
                    max_new_tokens=self.configs.test_pred_len,
                )
                predictions = outputs.reshape(B, -1, outputs.shape[-1])
                predictions = predictions.permute(0, 2, 1)
                y1, y2 = torch.chunk(predictions, 2, dim=0)
                predictions = (y1 - y2) / (self.a + self.b)
                predictions = predictions * stdev + means
                return predictions
        elif "sundial" in self.configs.model:
            if training:
                outputs = self.ltm(
                    input_ids=batch_x,
                    labels=batch_y,
                    pred_len=self.configs.test_pred_len,
                )
                losses = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss1, loss2 = torch.chunk(losses, 2, dim=0)
                loss1 = loss1.reshape(B, M, -1).permute(0, 2, 1).reshape(-1, M)
                loss2 = loss2.reshape(B, M, -1).permute(0, 2, 1).reshape(-1, M)

                with torch.no_grad():
                    outputs_origin = self.ltm(
                        input_ids=x0,
                        labels=y0,
                    )
                    loss_origin = outputs_origin["loss"] if isinstance(outputs, dict) else outputs[0]

                return loss1.mean() + loss2.mean() + torch.max((2 * (loss1 + loss2) / (self.a + self.b) ** 2).mean(), loss_origin.mean().detach())
            
            else:
                outputs = self.ltm.generate(
                    batch_x,
                    max_new_tokens=self.configs.test_pred_len,
                    num_samples=self.configs.test_n_sample,
                )
                predictions = outputs.reshape(
                    B, -1, outputs.shape[1], outputs.shape[-1]
                )
                predictions = predictions.permute(0, 2, 1, 3)  # B N C L
                predictions = predictions.permute(0, 1, 3, 2)  # B N L C
                y1, y2 = torch.chunk(predictions, 2, dim=0)
                predictions = (y1 - y2) / (self.a + self.b)
                predictions = predictions.permute(1, 0, 2, 3)
                predictions = predictions * stdev + means
                predictions = predictions.permute(1, 0, 2, 3)
                predictions = predictions.mean(dim=1)
                return predictions
        else:
            raise NotImplementedError
