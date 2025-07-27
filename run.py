import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from exp.exp_forecast_adaption import Exp_Forecast_Adaptation


def main():
    parser = argparse.ArgumentParser(description="Timer")

    # basic config
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
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
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
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
    parser.add_argument("--test_dir", type=str, default="./test", help="test dir")
    parser.add_argument(
        "--test_with_revin", action="store_true", help="test with revin", default=False
    )
    parser.add_argument("--test_n_sample", type=int, default=500, help="test n sample")

    parser.add_argument(
        "--adaptation", type=str, help="adaptation method description", default=None
    )
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
    parser.add_argument("--des", type=str, default="test", help="exp description")
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
    parser.add_argument(
        "--adapter", type=str, default="adapter_linear", help="adapter name"
    )
    parser.add_argument("--input_channel", type=int, default=1, help="input channel")
    parser.add_argument("--output_channel", type=int, default=1, help="output channel")
    parser.add_argument("--fit_batch_size", type=int, default=128)
    parser.add_argument("--fit_learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--use_amp", action="store_true", help="enable mixed precision training"
    )
    parser.add_argument("--accum_steps", type=int, default=32)
    parser.add_argument("--finetune_head_path", type=str, default='')
    parser.add_argument("--finetune_ltm", type=str, default='')
    args = parser.parse_args()

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

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

    Exp = Exp_Forecast_Adaptation

    if args.is_training:
        exp = Exp(args)  # set experiments
        setting = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            args.model,
            args.adapter,
            args.input_channel,
            args.output_channel,
            args.data_name,
            args.seq_len,
            args.test_pred_len,
            args.input_token_len,
            args.output_token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.accum_steps,
            args.test_n_sample,
        )

        if args.local_rank == 0:
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
        exp.train(setting)
        torch.cuda.empty_cache()
    else:
        setting = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            args.model,
            args.adapter,
            args.input_channel,
            args.output_channel,
            args.data_name,
            args.seq_len,
            args.test_pred_len,
            args.input_token_len,
            args.output_token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.accum_steps,
            args.test_n_sample,
        )
        exp = Exp(args)  # set experiments
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
