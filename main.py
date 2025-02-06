import argparse

import torch
import torchvision

from train import train


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.nn.functional.relu(
            torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2)
        )
        x = x.view(-1, 320)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x


def main(args):
    torch.manual_seed(args.seed)
    torch.multiprocessing.set_start_method("spawn", force=True)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    dataloader_kwargs = {"batch_size": args.batch_size, "shuffle": True}

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        dataloader_kwargs.update({"num_workers": 1, "pin_memory": True})
    elif args.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device} with {args.num_ranks} ranks")
    print(args)

    # Put the device onto the device, and, if on the CPU, into shared memory.
    # For CPU parallelisation, each process will be able to access (and update)
    # the model because it's in shared memory on the node. This will not work
    # for distributed parallelism, e.g. > 1 node or > 1 GPU.
    net = Net().to(device)
    if device.type in ["cpu", "cuda"]:  # MPS raises an exception
        net.share_memory()

    # Start each process running train()
    ranks = []
    for rank in range(args.num_ranks):
        p = torch.multiprocessing.Process(
            target=train,
            args=(rank, args, net, device, dataset, dataloader_kwargs),
        )
        p.start()
        ranks.append(p)

    # Use join to wait for all processes to finish
    for rank in ranks:
        rank.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mps", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num-ranks", default=1, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.5, type=float)
    args = parser.parse_args()

    main(args)
