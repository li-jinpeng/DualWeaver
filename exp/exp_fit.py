import numpy as np
import torch
import torch.nn.functional as F
import warnings
import os

from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from models import AdaPTS_prefit

warnings.filterwarnings("ignore")


class Exp_Fit(Exp_Basic):
    def __init__(self, args):
        super(Exp_Fit, self).__init__(args)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)

    def _build_model(self):
        model = AdaPTS_prefit.Model(self.args)
        self.device = torch.device("cuda:{}".format(self.args.local_rank))
        model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, fit=True)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.fit_learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        if self.args.local_rank == 0:
            print("next learning rate is {}".format(self.args.fit_learning_rate))
        return model_optim

    def train(self):
        train_set, train_loader = self._get_data("train")
        vali_set, vali_loader = self._get_data("val")
        adapter_optim = self._select_optimizer()

        if self.args.local_rank == 0:
            print("=> Fitting adapter layers...")

        best_loss = np.inf
        best_loss = torch.tensor(best_loss).to(self.device)

        for fit_epoch in range(300):
            self.model.train()
            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                loss = F.mse_loss(outputs, batch_x)

                adapter_optim.zero_grad()
                loss.backward()
                adapter_optim.step()

            self.model.eval()
            vali_loss = torch.tensor(0.0).to(self.device)
            vali_count = torch.tensor(0.0).to(self.device)
            with torch.no_grad():
                for i, (batch_x, batch_y, _, _) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    outputs = self.model(batch_x)
                    vali_loss += F.mse_loss(outputs, batch_x).item()
                    vali_count += batch_x.shape[0]

            dist.barrier()
            dist.reduce(vali_loss, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(vali_count, dst=0, op=dist.ReduceOp.SUM)

            this_loss = vali_loss / vali_count
            dist.barrier()
            dist.broadcast(this_loss, src=0)

            if self.args.local_rank == 0:
                print(f"Adapter Fit Epoch {fit_epoch+1}, Val Loss: {this_loss:.7f}")

            if this_loss < best_loss:
                best_loss = this_loss
                best_model = self.model
            else:
                break
        if self.args.local_rank == 0:
            if not os.path.exists("fit_model"):
                os.mkdir("fit_model")
            torch.save(
                {"model_state_dict": best_model.state_dict()},
                f"fit_model/{self.args.input_channel}_{self.args.output_channel}_{self.args.data_name}_checkpoint.pth",
            )
            print("Save Model!")
        dist.barrier()
