import os
import shutil
import time
import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import json

from copy import deepcopy

from transformers import AutoModelForCausalLM, AutoConfig
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from .exp_basic import Exp_Basic
from data_provider.data_factory import data_provider


class Exp_Forecast_Adaptation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast_Adaptation, self).__init__(args)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.args.use_amp)

    def _build_model(self):
        if not os.path.exists("metrics"):
            os.mkdir("metrics")
        self.device = torch.device("cuda:{}".format(self.args.local_rank))

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
            load_path = f"fit_model/{self.args.data_name}_checkpoint.pth"
            if not os.path.exists(load_path):
                if self.args.local_rank == 0:
                    print(f"Fit adapter layers for {self.args.data_name} first.")
                model = self._fit_for_AdaPTS(model)
            else:
                checkpoint = torch.load(
                    load_path,
                    map_location=self.device,
                )
                model.module.encoder.load_state_dict(checkpoint["encoder_state_dict"])
                model.module.decoder.load_state_dict(checkpoint["decoder_state_dict"])

        if self.args.adapter == "SFT":
            for name, param in model.named_parameters():
                param.requires_grad = True
        elif self.args.adapter == "ZeroShot":
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            metrics_file = f"metrics/{self.args.model}/{self.args.data_name}/SFT.json"
            with open(metrics_file, "r") as f:
                metrics = json.load(f)[str(self.args.test_pred_len)]
                best_lr = None
                best_mse = None
                for lr, metric in metrics.items():
                    mse, mae = metric.split(" ")
                    if best_lr is None or float(mse) < best_mse:
                        best_lr = lr
                        best_mse = float(mse)
            finetune_ltm_path = f"checkpoints/{self.args.model}/{self.args.data_name}/SFT/{best_lr}/model.pth"
            if self.args.local_rank == 0:
                print(f"Load finetuned LTM from {finetune_ltm_path}.")
            ltm_checkpoint = torch.load(
                finetune_ltm_path, map_location=self.device
            )
            model.module.ltm.load_state_dict(ltm_checkpoint["ltm_state_dict"])
            
            for name, param in model.named_parameters():
                if "ltm" not in name:
                    param.requires_grad = True
                    if self.args.local_rank == 0:
                        print(name)
                else:
                    param.requires_grad = False

        return model

    def _get_data(self, flag, fit=False):
        data_set, data_loader = data_provider(self.args, flag, fit=fit)
        return data_set, data_loader

    def _select_optimizer(self, fit=False, model=None):
        model_optim = optim.AdamW(
            self.model.parameters() if not fit else [p for n, p in model.named_parameters() if "encoder" in n or "decoder" in n],
            lr=self.args.learning_rate if not fit else self.args.fit_learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        return model_optim
    
    def _fit_for_AdaPTS(self, model):
        train_set, train_loader = self._get_data("train", fit=True)
        vali_set, vali_loader = self._get_data("val", fit=True)

        adapter_optim = self._select_optimizer(fit=True, model=model)
        if self.args.local_rank == 0:
            print("next learning rate is {}".format(self.args.fit_learning_rate))
            print("=> Fitting adapter layers...")
        
        best_loss = np.inf
        best_loss = torch.tensor(best_loss).to(self.device)

        for fit_epoch in range(300):
            model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = model.module.encoder(batch_x)
                outputs = model.module.decoder(outputs)
                loss = F.mse_loss(outputs, batch_x)

                adapter_optim.zero_grad()
                loss.backward()
                adapter_optim.step()
            dist.barrier()

            model.eval()
            vali_loss = torch.tensor(0.0).to(self.device)
            vali_count = torch.tensor(0.0).to(self.device)
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    outputs = model.module.encoder(batch_x)
                    outputs = model.module.decoder(outputs)

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
                best_model = model
            else:
                break
        if self.args.local_rank == 0:
            if not os.path.exists("fit_model"):
                os.mkdir("fit_model")
            torch.save(
                {"encoder_state_dict": best_model.module.encoder.state_dict(),
                 "decoder_state_dict": best_model.module.decoder.state_dict()},
                f"fit_model/{self.args.data_name}_checkpoint.pth",
            )
            print("Save Model!")
        dist.barrier()
        return model
        

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
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                with torch.amp.autocast("cuda", enabled=False):
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
                        self.args.adapter == "SFT"
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
                print(
                    f"{self.args.adapter} {self.args.model} {self.args.data_name} epoch: {epoch+1} {end-start}"
                )

        total_mse_loss = total_mse_loss.item() / total_count.item()
        total_mae_loss = total_mae_loss.item() / total_count.item()
        self.model.train()

        return total_mse_loss, total_mae_loss

    def train(self):
        if not os.path.exists("metrics"):
            os.mkdir("metrics")

        train_data, train_loader = self._get_data(flag="train")

        path = f"checkpoints/{self.args.model}/{self.args.data_name}/{self.args.adapter}/{self.args.learning_rate:.6f}"
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
            for i, (batch_x, batch_y) in enumerate(train_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                with torch.amp.autocast("cuda", enabled=self.args.use_amp):
                    loss = self.model(batch_x, batch_y)
                    if isinstance(loss, tuple):
                        loss, re_flag = loss
                        saved_re_dir = f"./re_flag/{self.args.model}/{self.args.data_name}/{self.args.adapter}/{self.args.test_pred_len}/{self.args.learning_rate:.6f}/{epoch}"
                        os.makedirs(saved_re_dir, exist_ok=True)
                        torch.save(re_flag, os.path.join(saved_re_dir, f"{self.args.local_rank}_{i}.pt"))    
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
                print(
                    f"{self.args.adapter} {self.args.model} {self.args.data_name} {end-start}"
                )
                print(
                    f"{self.args.adapter} {self.args.model} {self.args.data_name} {peak_mem / 1e9:.2f} GB"
                )

            scheduler.step()

            (
                vali_mse_loss,
                _,
                test_mse_loss,
                test_mae_loss,
            ) = self.test(epoch=epoch)

            torch.cuda.empty_cache()
            if dist.get_rank() == 0:
                print(f"best_loss: {best_loss}; this_loss: {vali_mse_loss}")
                if vali_mse_loss < best_loss:
                    for epoch_idx in range(self.args.train_epochs):
                        saved_re_dir = f"./re_flag/{self.args.model}/{self.args.data_name}/{self.args.adapter}/{self.args.test_pred_len}/{self.args.learning_rate:.6f}/{epoch_idx}"
                        if os.path.exists(saved_re_dir) and epoch_idx != epoch:
                            shutil.rmtree(saved_re_dir)
                    result_path =  f"metrics/{self.args.model}/{self.args.data_name}/{self.args.adapter}.json"
                    os.makedirs(os.path.dirname(result_path), exist_ok=True)
                    result_key = f"{self.args.test_pred_len}"
                    result_key_param = f"{self.args.learning_rate:.6f}"
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
                    if self.args.adapter == "SFT":
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
                    saved_re_dir = f"./re_flag/{self.args.model}/{self.args.data_name}/{self.args.adapter}/{self.args.test_pred_len}/{self.args.learning_rate:.6f}/{epoch}"
                    if os.path.exists(saved_re_dir):
                        shutil.rmtree(saved_re_dir)
                    patience -= 1
                    print("Result not saved!")
                    print(f"Patience: {patience.item()}")
            dist.barrier()
            dist.broadcast(patience, src=0)
            if patience < 0:
                if dist.get_rank() == 0:
                    print("Early stop!")
                break

    def test(self, epoch=0):
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
            print("=============Test=============")
            print("Dataset: MSE, MAE")

        vali_mse_loss, vali_mae_loss = self.vali(vali_loader, "vali", epoch=epoch)
        test_mse_loss, test_mae_loss = self.vali(test_loader, "test", epoch=epoch)
        if self.args.adapter == "ZeroShot" and self.args.local_rank == 0:
            result_path =  f"metrics/{self.args.model}/{self.args.data_name}/{self.args.adapter}.json"
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            result_key = f"{self.args.test_pred_len}"
            if os.path.exists(result_path):
                with open(result_path, "r") as result_file:
                    try:
                        result_content = json.load(result_file)
                    except:
                        result_content = {}
            else:
                result_content = {}
            result_content[result_key] = f"{test_mse_loss} {test_mae_loss}"
            with open(result_path, "w") as result_file:
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
