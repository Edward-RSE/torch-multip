import argparse
import os
import socket

import torch

from model import Net
from torch_multip.train import train_model
from torch_multip.validate import validate_model


def create_world(rank: int, world_size: int, backend: str | None = None) -> int:
    """Initialise the world distributed training group.

    Parameters
    ----------
    rank: int
        The rank (process_id) of the current process
    world_size: int
        The number of processes being initialised.
    backend: str
        The backend to use for communication.

    Returns
    -------
    int
        The global rank of the current process
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.distributed.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
    )

    return torch.distributed.get_rank()


def destroy_world() -> None:
    """Destroy the world process group."""
    torch.distributed.destroy_process_group()


def initialise_device(local_rank: int, args: argparse.Namespace) -> torch.device:
    """Initialise the device used by the rank.

    If using CUDA, the local rank, set by torchrun, is used to determine the
    device. If torchrun is not being used, then the rank passed will be used.

    Parameters
    ----------
    rank: int
        The (local) rank of the current process.
    args: argparse.Namespace
        The command line arguments.

    Returns
    -------
    torch.device
        The device assigned to the rank
    """
    if args.use_cuda:
        local_world_size = int(
            os.environ.setdefault("LOCAL_WORLD_SIZE", str(args.world_size))
        )
        if local_world_size > torch.cuda.device_count():
            hostname = socket.gethostname()
            raise RuntimeError(
                f"Local world size of {local_world_size} on {hostname} is greater than device count of {torch.cuda.device_count()}"
            )
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return device


def initialise_distributed_dataloader(
    rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
    dataloader_kwargs: dict,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DistributedSampler]:
    """Create a data loader and sampler for distributed training.

    Parameters
    ----------
    rank: int
        The calling rank
    world_size: int
        The total number of ranks
    dataset: torch.utils.data.Dataset
        The dataset to load
    dataloader_kwargs: dict
        The keyword arguments to pass to the data loader

    Returns
    -------
    torch.utils.data.DataLoader
        The data loader
    torch.utils.data.DistributedSampler
        The distributed sampler
    """
    dataloader_kwargs["shuffle"] = None
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler, **dataloader_kwargs
    ), sampler


def initialise_validation_dataloader(
    dataset: torch.utils.data.Dataset, dataloader_kwargs: dict
) -> torch.utils.data.DataLoader:
    """Initialise the data loader for validation (not distributed)

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        The dataset to load
    dataloader_kwargs: dict
        The keyword arguments to pass to the data loader

    Returns
    -------
    torch.utils.data.DataLoader
        The data loader
    """
    dataloader_kwargs["shuffle"] = True
    return torch.utils.data.DataLoader(
        dataset,
        **dataloader_kwargs,
    )


def distributed_worker(
    proc_id: str,
    args: argparse.Namespace,
    dataset: torch.utils.data.Dataset,
    dataloader_kwargs: dict,
) -> None:
    """Initialise and train a model using distributed parallelism.

    This function sets up the distributed environment (including distributed
    models and dataloaders), trains the model and then validates the model.

    Parameters
    ----------
    prod_id  : str
        The ID of the spawned process
    args : argparse.Namespace
        The command line arguments for the program
    dataset : torch.utils.data.Dataset
        The dataset to train the model on
    dataloader_kwargs : dict
        The keyword arguments to pass to the data loader
    """
    # Set the initial rank using either the LOCAL_RANK variable if using
    # torchrun, or the process id from torch.multiprocessing.spawn
    local_rank = int(os.environ.setdefault("LOCAL_RANK", str(proc_id)))

    # Returns rank either from LOCAL_RANK or from torch.distributed.get_rank(),
    # which will be the (global) rank after initialisation of the world group
    global_rank = create_world(
        local_rank, args.world_size, backend="nccl" if args.use_cuda else "gloo"
    )
    device = initialise_device(local_rank, args)
    print(f"Global rank {global_rank} created using local device {device}")

    model = Net().to(device)
    distributed_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device] if args.use_cuda else None,
        output_device=device if args.use_cuda else None,
    )

    distributed_dataloader, distributed_sampler = initialise_distributed_dataloader(
        global_rank, args.world_size, dataset, dataloader_kwargs
    )

    train_model(
        global_rank,
        args,
        distributed_model,
        device,
        distributed_dataloader,
        sampler=distributed_sampler,
    )
    torch.distributed.barrier()

    if global_rank == 0:
        torch.save(distributed_model.state_dict(), "model.pt")
        validation_dataloader = initialise_validation_dataloader(
            dataset, dataloader_kwargs
        )
        validate_model(args, distributed_model, device, validation_dataloader)

    torch.distributed.barrier()
    destroy_world()
    print(f"Rank {global_rank} has finished")


def train_distributed(
    args: argparse.Namespace, dataset: torch.utils.data.Dataset, dataloader_kwargs: dict
) -> None:
    """Train the model using distributed parallelism

    This function spawns processes to train the model.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments for the program
    dataset : torch.utils.data.Dataset
        The dataset to train the model on
    dataloader_kwargs : dict
        The keyword arguments to pass to the data loader
    """
    args.world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    torch.multiprocessing.spawn(
        distributed_worker,
        args=(args, dataset, dataloader_kwargs),
        nprocs=1,
        join=True,
    )
