import argparse

import torch
import torchvision

from torch_multip.distributed import train_distributed
from torch_multip.multiprocess import train_multiprocess


def main(args):
    print(args)
    torch.manual_seed(args.seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
    }

    if args.mode == "distributed":
        train_distributed(args, dataset, dataloader_kwargs)
    else:
        train_multiprocess(args, dataset, dataloader_kwargs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["distributed", "multiprocess"])
    parser.add_argument("--dist-backend", default="gloo", type=str)
    parser.add_argument("--use-mps", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-epochs", default=15, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.5, type=float)
    args = parser.parse_args()

    main(args)
