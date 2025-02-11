import os

import torch

from model import Net
from torch_multip.train import train_model
from torch_multip.validate import validate_model


def create_world(rank, world_size, backend):
    torch.distributed.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
    )


def destroy_world():
    torch.distributed.destroy_process_group()


def distributed_worker(rank, args, dataset, dataloader_kwargs):
    create_world(rank, args.world_size, args.dist_backend)
    print(f"Rank {rank} initialised")

    if torch.cuda.is_available() and args.use_cuda:
        device_id = torch.distributed.get_rank() % torch.cuda.device_count()
        device = torch.device(f"cuda:{device_id}")
        device_ids = [device_id]
    else:
        device = torch.device("cpu")
        device_ids = None

    model = Net().to(device)
    distributed_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids
    )

    train_model(rank, args, distributed_model, device, dataset, dataloader_kwargs)

    torch.distributed.barrier()
    if rank == 0:
        torch.save(model.state_dict(), "model.pt")

    destroy_world()
    print(f"Rank {rank} finished")


def train_distributed(args, dataset, dataloader_kwargs):
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.multiprocessing.spawn(
        distributed_worker,
        args=(args, dataset, dataloader_kwargs),
        nprocs=args.world_size,
        join=True,
    )

    model = Net()
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    validate_model(args, model, "cpu", dataset, dataloader_kwargs)
