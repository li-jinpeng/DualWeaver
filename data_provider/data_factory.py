import math
import torch

from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from data_provider.data_loader import (
    MultivariateDatasetBenchmark,
    FinetuneDatasetBenchmark,
)


class ValDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        indices = self.indices
        start = self.rank * self.num_samples
        end = min(start + self.num_samples, len(self.indices))
        return iter(indices[start:end])

    def __len__(self):
        return min(self.num_samples, len(self.dataset) - self.rank * self.num_samples)


def data_provider(args, flag, fit=False):
    if "Finetune" in args.adapter or args.adapter == "ZeroShot":
        Data = FinetuneDatasetBenchmark
    else:
        Data = MultivariateDatasetBenchmark

    if flag == "train":
        dataset = Data(
            seq_len=args.seq_len,
            input_token_len=args.input_token_len,
            output_token_len=args.output_token_len,
            pred_len=args.test_pred_len,
            scale=True,
            data_path=args.data_path,
            flag=flag,
        )

        train_datasampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size if not fit else args.fit_batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=False,
            pin_memory=True,
            drop_last=False,
        )
    else:

        dataset = Data(
            seq_len=args.seq_len,
            input_token_len=args.input_token_len,
            output_token_len=args.output_token_len,
            pred_len=args.test_pred_len,
            scale=True,
            data_path=args.data_path,
            flag=flag,
        )

        val_datasampler = ValDistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size if not fit else args.fit_batch_size,
            sampler=val_datasampler,
            num_workers=args.num_workers,
            drop_last=False,
        )

    return dataset, data_loader
