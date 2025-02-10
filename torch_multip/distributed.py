import os

import torch

from model import Net
from torch_multip.train import train_model
from torch_multip.validate import validate_model


def setup(rank, world_size, backend):
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12356"
    torch.distributed.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    torch.distributed.destroy_process_group()


def _train_distributed(rank, args, dataset, dataloader_kwargs):
    setup(rank, args.world_size, args.dist_backend)
    print("Rank", torch.distributed.get_rank(), "is alive")
    model = Net()
    distributed_model = torch.nn.parallel.DistributedDataParallel(
        model,  # device_ids=[rank]
    )
    train_model(rank, args, distributed_model, "cpu", dataset, dataloader_kwargs)
    validate_model(args, model, "cpu", dataset, dataloader_kwargs)

    cleanup()
    return


def train_distributed(args, dataset, dataloader_kwargs):
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.multiprocessing.spawn(
        _train_distributed,
        args=(args, dataset, dataloader_kwargs),
        nprocs=args.world_size,
        join=True,
    )
