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
    local_rank = int(os.environ.setdefault("LOCAL_RANK", str(rank)))
    if args.use_cuda:
        if local_rank > torch.cuda.device_count() - 1:
            raise RuntimeError(
                f"Local rank {local_rank} is greater than device count {torch.cuda.device_count()} on current node"
            )
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return device


def initialise_distributed_dataloader(rank, world_size, dataset, dataloader_kwargs):
    dataloader_kwargs["shuffle"] = None
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler, **dataloader_kwargs
    ), sampler


def initialise_validation_dataloader(dataset, dataloader_kwargs):
    dataloader_kwargs["shuffle"] = True
    return torch.utils.data.DataLoader(
        dataset,
        **dataloader_kwargs,
    )


def distributed_worker(rank, args, dataset, dataloader_kwargs):
    create_world(rank, args.world_size, args.dist_backend)
    device = initialise_device(rank, args)
    print(f"Rank {rank} created using device {device}")

    model = Net().to(device)
    distributed_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank] if args.use_cuda else None
    )

    distributed_dataloader, distributed_sampler = initialise_distributed_dataloader(
        rank, args.world_size, dataset, dataloader_kwargs
    )

    train_model(
        rank,
        args,
        distributed_model,
        device,
        distributed_dataloader,
        sampler=distributed_sampler,
    )
    torch.distributed.barrier()

    if rank == 0:
        torch.save(distributed_model.state_dict(), "model.pt")
        validation_dataloader = initialise_validation_dataloader(
            dataset, dataloader_kwargs
        )
        validate_model(args, distributed_model, device, validation_dataloader)

    torch.distributed.barrier()
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
