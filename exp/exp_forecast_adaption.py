import os
import time
import torch
import warnings
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import json

from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoConfig
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from .exp_basic import Exp_Basic
from .exp_fit import Exp_Fit
from data_provider.data_factory import data_provider

warnings.filterwarnings("ignore")


class Exp_Forecast_Adaptation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast_Adaptation, self).__init__(args)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)

    def _build_model(self):
        if not os.path.exists("metrics"):
            os.mkdir("metrics")
        self.device = torch.device("cuda:{}".format(self.args.local_rank))
        if self.args.adapter == "AdaPTS":
            load_path = f"fit_model/{self.args.input_channel}_{self.args.output_channel}_{self.args.data_name}_checkpoint.pth"
            if not os.path.exists(load_path):
                exp_fit = Exp_Fit(self.args)
                start = time.time()

                exp_fit.train()

                torch.cuda.synchronize()
                dist.barrier()
                end = time.time()
                if self.args.local_rank == 0:
                    print(f"{self.args.adapter} {self.args.data_name} {self.args.output_channel} {end-start}")

            checkpoint = torch.load(
                load_path,
                map_location=self.device,
            )
        if "timer" in self.args.model or "sundial" in self.args.model:
            config = AutoConfig.from_pretrained(
                self.args.pretrained_model_path, trust_remote_code=True
            )
            ltm = AutoModelForCausalLM.from_pretrained(
                self.args.pretrained_model_path, trust_remote_code=True, config=config
            )
        else:
            raise NotImplementedError

        model = self.adapter_dict[self.args.adapter].Model(self.args, ltm)
        model = DDP(
            model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True
        )
        model = model.to(self.device)

        if self.args.adapter == "AdaPTS":
            encoder_state_dict = OrderedDict()
            decoder_state_dict = OrderedDict()
            for k, v in checkpoint["model_state_dict"].items():
                if k.startswith("module."):
                    k = k.replace("module.", "", 1)  # 移除第一个 'module.'
                if k.startswith("encoder."):
                    encoder_state_dict[k.replace("encoder.", "", 1)] = v
                if k.startswith("decoder."):
                    decoder_state_dict[k.replace("decoder.", "", 1)] = v
            model.module.encoder.load_state_dict(encoder_state_dict)
            model.module.decoder.load_state_dict(decoder_state_dict)

        if self.args.model == "timerxl" and self.args.finetune_head_path != "":
            lm_heads_checkpoint = torch.load(
                self.args.finetune_head_path, map_location=self.device
            )
            if self.args.local_rank == 0:
                print(f"Load ltm heads from {self.args.finetune_head_path}.")
            model.module.ltm.lm_heads.load_state_dict(lm_heads_checkpoint["lm_head"])

        if self.args.model == "timerxl" and self.args.finetune_ltm != "":
            ltm_checkpoint = torch.load(
                self.args.finetune_ltm, map_location=self.device
            )
            if self.args.local_rank == 0:
                print(f"Load ltm from {self.args.finetune_ltm}.")
            ltm_state_dict = OrderedDict()
            for k, v in ltm_checkpoint["ltm_state_dict"].items():
                if k.startswith("ltm."):
                    ltm_state_dict[k.replace("ltm.", "", 1)] = v
            model.module.ltm.load_state_dict(ltm_checkpoint["ltm_state_dict"])

        if "Finetune" in self.args.adapter:
            if "linear_probing" in self.args.adapter:
                for name, param in model.named_parameters():
                    if "ltm.lm_head" in name or "ltm.flow_loss" in name:
                        param.requires_grad = True
                        if self.args.local_rank == 0:
                            print(name)
                    else:
                        param.requires_grad = False
            else:
                for name, param in model.named_parameters():
                    param.requires_grad = True
        elif "ZeroShot" in self.args.adapter:
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if "ltm" not in name:
                    param.requires_grad = True
                    if self.args.local_rank == 0:
                        print(name)
                else:
                    param.requires_grad = False

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        return model_optim

    def vali(self, vali_loader, flag, epoch=0, chunk_size=100):
        if self.args.local_rank == 0:
            print(f"Start {flag}...")
        total_mse_loss = torch.tensor(0.0).to(self.device)
        total_mae_loss = torch.tensor(0.0).to(self.device)
        total_count = torch.tensor(0.0).to(self.device)
        iter_count = 0
        time_now = time.time()
        test_steps = len(vali_loader)
        self.model.eval()
        with torch.no_grad():
            chunk_mse_loss = torch.tensor(0.0).to(self.device)
            chunk_mae_loss = torch.tensor(0.0).to(self.device)
            chunk_count = torch.tensor(0.0).to(self.device)

            start = time.time()
            for i, (batch_x, batch_y, _, _) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    pred = self.model(batch_x)

                    mae_loss = (
                        F.l1_loss(pred, batch_y, reduction="none").mean(dim=1).sum()
                    )
                    mse_loss = (
                        F.mse_loss(pred, batch_y, reduction="none").mean(dim=1).sum()
                    )

                    chunk_mse_loss += mse_loss
                    chunk_mae_loss += mae_loss
                    if (
                        "Finetune" in self.args.adapter
                        or self.args.adapter == "ZeroShot"
                    ):
                        chunk_count += batch_x.shape[0]
                    else:
                        chunk_count += batch_x.shape[0] * batch_x.shape[-1]

                if (i + 1) % chunk_size == 0 or (i + 1) == len(vali_loader):
                    dist.barrier()
                    dist.reduce(chunk_mse_loss, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(chunk_mae_loss, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(chunk_count, dst=0, op=dist.ReduceOp.SUM)

                    # Accumulate global loss
                    total_mse_loss += chunk_mse_loss
                    total_mae_loss += chunk_mae_loss
                    total_count += chunk_count

                    # Reset chunk metrics
                    chunk_mse_loss.zero_()
                    chunk_mae_loss.zero_()
                    chunk_mse_loss.zero_()
                    chunk_count.zero_()

                    if self.args.local_rank == 0:
                        speed = (time.time() - time_now) / (i + 1)
                        left_time = speed * (test_steps - i)
                        print(
                            "\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(
                                i + 1, speed, left_time
                            )
                        )
            torch.cuda.synchronize()
            dist.barrier()
            end = time.time()
            if self.args.local_rank == 0:
                print(f"{self.args.adapter} {self.args.model} {self.args.data_name} {self.args.output_channel} epoch: {epoch+1} {end-start}")

        total_mse_loss = total_mse_loss.item() / total_count.item()
        total_mae_loss = total_mae_loss.item() / total_count.item()
        self.model.train()

        return total_mse_loss, total_mae_loss

    def train(self, setting):
        if not os.path.exists("metrics"):
            os.mkdir("metrics")

        train_data, train_loader = self._get_data(flag="train")

        path = os.path.join(self.args.checkpoints, setting)
        if self.args.local_rank == 0:
            os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim,
            T_max=self.args.train_epochs,
            eta_min=1e-8,
        )
        accum_steps = self.args.accum_steps
        total_steps = 0
        start_epoch = 0
        best_loss = np.inf
        patience = torch.tensor(3).to(self.device)

        for epoch in range(start_epoch, self.args.train_epochs):
            start = time.time()
            if self.args.local_rank == 0:
                print(f"Epoch {epoch+1}/{self.args.train_epochs}")
            train_loader.sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = 0.0
            for i, (batch_x, batch_y, loss_mask, y_mask) in enumerate(train_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                loss_mask = loss_mask.float().to(self.device)
                y_mask = y_mask.float().to(self.device)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    loss = self.model(batch_x, batch_y, loss_mask, y_mask, training=1)
                    loss /= accum_steps

                self.scaler.scale(loss).backward()
                epoch_loss += loss.item() * accum_steps

                if (i + 1) % accum_steps == 0 or i + 1 == len(train_loader):
                    self.scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.scaler.step(model_optim)
                    self.scaler.update()
                    model_optim.zero_grad()
                    model_optim.step()
                    total_steps += 1

                    if total_steps % 10 == 0 and self.args.local_rank == 0:
                        avg_loss = epoch_loss / (accum_steps * 10)
                        print(
                            f"Step {total_steps}: loss={avg_loss:.7f}, lr={model_optim.param_groups[0]['lr']:.10f}"
                        )
                        epoch_loss = 0.0

            torch.cuda.synchronize()
            dist.barrier()
            end = time.time()
            peak_mem = torch.cuda.max_memory_allocated()
            if self.args.local_rank == 0:
                print(f"{self.args.adapter} {self.args.model} {self.args.data_name} {self.args.output_channel} {end-start}")
                print(f"{self.args.adapter} {self.args.model} {self.args.data_name} {self.args.output_channel} {peak_mem / 1e9:.2f} GB")

            scheduler.step()

            (
                vali_mse_loss,
                _,
                test_mse_loss,
                test_mae_loss,
            ) = self.test(setting, epoch=epoch)

            torch.cuda.empty_cache()
            if dist.get_rank() == 0:
                print(f"best_loss: {best_loss}; this_loss: {vali_mse_loss}")
                if vali_mse_loss < best_loss:
                    if self.args.model == "timerxl" and (
                        self.args.finetune_head_path != ""
                        or self.args.finetune_ltm != ""
                    ):
                        result_path = f"metrics/{self.args.model}_{self.args.adapter}_{self.args.data_name}_with_LP.json"
                    else:
                        result_path = f"metrics/{self.args.model}_{self.args.adapter}_{self.args.data_name}.json"
                    result_key = f"{self.args.model}_{self.args.adapter}_{self.args.test_pred_len}"
                    result_key_param = (
                        f"{self.args.learning_rate}_{self.args.output_channel}"
                    )
                    if not os.path.exists(result_path):
                        result_content = {}
                        result_content[result_key] = {
                            result_key_param: f"{test_mse_loss} {test_mae_loss}"
                        }
                    else:
                        with open(result_path, "r") as result_file:
                            try:
                                result_content = json.load(result_file)
                            except:
                                result_content = {}
                            if result_key in result_content:
                                result_content[result_key][
                                    result_key_param
                                ] = f"{test_mse_loss} {test_mae_loss}"
                            else:
                                result_content[result_key] = {
                                    result_key_param: f"{test_mse_loss} {test_mae_loss}"
                                }
                    with open(result_path, "w") as result_file:
                        json.dump(result_content, result_file, indent=4)
                    best_loss = vali_mse_loss
                    if self.args.adapter == "Finetune_full":
                        torch.save(
                            {
                                "ltm_state_dict": self.model.module.ltm.state_dict(),
                            },
                            f"{path}/model.pth",
                        )
                    else:
                        if self.args.adapter == "AdaPTS":
                            torch.save(
                                {
                                    "encoder_state_dict": self.model.module.encoder.state_dict(),
                                    "decoder_state_dict": self.model.module.decoder.state_dict(),
                                },
                                f"{path}/model.pth",
                            )
                        elif self.args.adapter == "Finetune_linear_probing":
                            if "timer" in self.args.model:
                                torch.save(
                                    {
                                        "lm_head": self.model.module.ltm.lm_heads.state_dict(),
                                    },
                                    f"{path}/model.pth",
                                )
                            else:
                                torch.save(
                                    {
                                        "flow_loss": self.model.module.ltm.flow_loss.state_dict(),
                                    },
                                    f"{path}/model.pth",
                                )
                        else:
                            torch.save(
                                {
                                    "feature_weaver_state_dict": self.model.module.feature_weaver.state_dict(),
                                    "a": self.model.module.a,
                                    "b": self.model.module.b,
                                },
                                f"{path}/model.pth",
                            )
                    print("Result saved")
                    patience = torch.tensor(3).to(self.device)
                    print(f"Patience: {patience.item()}")
                else:
                    patience -= 1
                    print("Result not saved!")
                    print(f"Patience: {patience.item()}")
            dist.barrier()
            dist.broadcast(patience, src=0)
            if patience < 0:
                if dist.get_rank() == 0:
                    print("Early stop!")
                break

    def test(self, setting, epoch=0, test=0):
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        if self.args.local_rank == 0:
            print(
                "info:",
                self.args.input_token_len,
                self.args.output_token_len,
                self.args.test_pred_len,
            )

        if self.args.local_rank == 0:
            print(f"=============Test=============")
            print("Dataset: MSE, MAE")

        vali_mse_loss, vali_mae_loss = self.vali(vali_loader, "vali", epoch=epoch)
        test_mse_loss, test_mae_loss = self.vali(test_loader, "test", epoch=epoch)
        if self.args.adapter == "ZeroShot" and self.args.local_rank == 0:
            result_key = f"{self.args.model}_zero_shot_{self.args.seq_len}_{self.args.test_pred_len}_{self.args.data_name}"
            if os.path.exists("metrics/zero_shot.json"):
                with open("metrics/zero_shot.json", "r") as result_file:
                    try:
                        result_content = json.load(result_file)
                    except:
                        result_content = {}
            else:
                result_content = {}
            result_content[result_key] = f"{test_mse_loss} {test_mae_loss}"
            with open("metrics/zero_shot.json", "w") as result_file:
                json.dump(result_content, result_file, indent=4)
        if self.args.local_rank == 0:
            print(
                "vali loss mse mae: {:.7f}, {:.7f}".format(vali_mse_loss, vali_mae_loss)
            )
            print(
                "test loss mse mae: {:.7f}, {:.7f}".format(test_mse_loss, test_mae_loss)
            )

        return (
            vali_mse_loss,
            vali_mae_loss,
            test_mse_loss,
            test_mae_loss,
        )
