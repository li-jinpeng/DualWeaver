import os
import argparse
import random
import json
import numpy as np
import torch
import torch.distributed as dist
from copy import deepcopy
from exp.exp_forecast_adaption import Exp_Forecast_Adaptation


def main():
    parser = argparse.ArgumentParser(description="Timer")

    # basic config
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="timer",
        help="model name, options: [timer]",
    )
    parser.add_argument("--seed", type=int, default=2021, help="seed")

    # data loader
    parser.add_argument("--data_name", type=str, required=True, help="dataset name")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--scale", action="store_true", help="scale data")

    # forecasting task
    parser.add_argument(
        "--seq_len", type=int, default=672, help="input sequence length"
    )
    parser.add_argument(
        "--input_token_len", type=int, default=576, help="input token length"
    )
    parser.add_argument(
        "--output_token_len", type=int, default=96, help="max output token length"
    )

    # test
    parser.add_argument("--test_pred_len", type=int, default=96, help="test pred len")
    parser.add_argument("--test_n_sample", type=int, default=500, help="test n sample")

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="pretrain_model.pth",
        help="pretrain model path",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=32, help="data loader num workers"
    )
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0)
    # GPU
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--ddp", action="store_true", help="Distributed Data Parallel", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    )
    # adaptation
    parser.add_argument("--adapter", type=str, default="ZeroShot", help="adapter name")
    parser.add_argument("--input_channel", type=int, default=1, help="input channel")
    parser.add_argument("--fit_batch_size", type=int, default=128)
    parser.add_argument("--fit_learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--use_amp", action="store_true", help="enable mixed precision training"
    )
    parser.add_argument("--accum_steps", type=int, default=32)
    args = parser.parse_args()

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    metrics_file = f"metrics/{args.model}/{args.data_name}/{args.adapter}.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        if str(args.test_pred_len) in metrics:
            if (args.adapter == "ZeroShot"
                or f"{args.learning_rate:.6f}" in metrics[str(args.test_pred_len)]
            ):
                print(
                    f"Experiment for {args.model} on {args.data_name} with adapter {args.adapter} and pred len {args.test_pred_len} and lr {args.learning_rate} already done. Skipping..."
                )
                return

    if args.ddp:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        args.local_rank = local_rank
        if args.local_rank == 0:
            print(ip, port, hosts, rank, local_rank, gpus)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{ip}:{port}",
            world_size=hosts,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
    else:
        args.local_rank = 0

    if args.adapter != "SFT" and args.adapter != "ZeroShot":
        metrics_file = f"metrics/{args.model}/{args.data_name}/SFT.json"
        if not os.path.exists(metrics_file) or str(args.test_pred_len) not in json.load(
            open(metrics_file, "r")
        ):
            sft_args = deepcopy(args)
            sft_args.adapter = "SFT"
            sft_args.batch_size = 256
            sft_args.accum_steps = 1
            for lr in [5e-5, 1e-5, 5e-6, 1e-6]:
                sft_args.learning_rate = lr
                exp = Exp_Forecast_Adaptation(sft_args)
                exp.train()
        else:
            metrics = json.load(open(metrics_file, "r"))[str(args.test_pred_len)]
            for lr in [5e-5, 1e-5, 5e-6, 1e-6]:
                if f"{lr:.6f}" not in metrics:
                    sft_args = deepcopy(args)
                    sft_args.adapter = "SFT"
                    sft_args.batch_size = 256
                    sft_args.accum_steps = 1
                    sft_args.learning_rate = lr
                    exp = Exp_Forecast_Adaptation(sft_args)
                    exp.train()
    else:
        args.batch_size = 256
        args.accum_steps = 1

    if args.adapter == "ZeroShot":
        exp = Exp_Forecast_Adaptation(args)
        exp.test()
        torch.cuda.empty_cache()
        return

    exp = Exp_Forecast_Adaptation(args)
    exp.train()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
