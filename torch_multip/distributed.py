import os

import torch

from model import Net
from torch_multip.train import train_model
from torch_multip.validate import validate_model


def create_world(rank, world_size, backend):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.distributed.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
    )


def destroy_world():
    torch.distributed.destroy_process_group()


def initialise_device(rank, args):
    if args.use_cuda:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    return device


def initialise_dataloader(rank, world_size, dataset, dataloader_kwargs):
    dataloader_kwargs["shuffle"] = None
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler, **dataloader_kwargs
    ), sampler


def distributed_worker(rank, args, dataset, dataloader_kwargs):
    create_world(rank, args.world_size, args.dist_backend)
    device = initialise_device(rank, args)
    print(f"Rank {rank} created using device {device}")

    model = Net().to(device)
    distributed_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank] if args.use_cuda else None
    )

    dataloader, sampler = initialise_dataloader(
        rank, args.world_size, dataset, dataloader_kwargs
    )

    train_model(
        rank,
        args,
        distributed_model,
        device,
        dataloader,
        sampler=sampler,
    )

    torch.distributed.barrier()
    if rank == 0:
        torch.save(model.state_dict(), "model.pt")

    destroy_world()
    print(f"Rank {rank} has finished")


def train_distributed(args, dataset, dataloader_kwargs):
    args.world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    torch.multiprocessing.spawn(
        distributed_worker,
        args=(args, dataset, dataloader_kwargs),
        nprocs=args.world_size,
        join=True,
    )

    model = Net()
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    validate_model(args, model, "cpu", dataset, dataloader_kwargs)
