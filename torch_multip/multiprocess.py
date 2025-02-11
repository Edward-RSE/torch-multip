import torch

from model import Net
from torch_multip.train import train_model
from torch_multip.validate import validate_model


def train_multiprocess(args, dataset, dataloader_kwargs):
    torch.multiprocessing.set_start_method("spawn", force=True)
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        dataloader_kwargs.update({"num_workers": 1, "pin_memory": True})
    elif args.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Put the device onto the device, and, if on the CPU, into shared memory.
    # For CPU parallelisation, each process will be able to access (and update)
    # the model because it's in shared memory on the node. This will not work
    # for distributed parallelism, e.g. > 1 node or > 1 GPU.
    model = Net().to(device)
    model.share_memory()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        **dataloader_kwargs,
    )

    # Start each process running train()
    ranks = []
    for rank in range(args.world_size):
        p = torch.multiprocessing.Process(
            target=train_model,
            args=(rank, args, model, device, dataloader),
        )
        p.start()
        ranks.append(p)

    # Use join to wait for all processes to finish
    for rank in ranks:
        rank.join()

    validate_model(args, model, device, dataloader)
