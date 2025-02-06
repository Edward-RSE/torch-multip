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

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dataloader_kwargs.update({"num_workers": 1, "pin_memory": True})
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    net = Net().to(device)
    net.share_memory()

    ranks = []
    for rank in range(args.num_ranks):
        p = torch.multiprocessing.Process(
            target=train, args=(rank, args, net, device, dataset, dataloader_kwargs)
        )
        p.start()
        ranks.append(p)

    for rank in ranks:
        rank.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_ranks", default=1, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.5, type=float)
    parser.add_argument("--log_interval", default=10, type=int)
    args = parser.parse_args()

    main(args)
