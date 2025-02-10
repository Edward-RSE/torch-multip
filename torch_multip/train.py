import time

import torch
import torch.utils.data.dataloader


def train_model(rank, args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)
    data_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    start = time.time()

    for epoch in range(1, args.num_epochs + 1):
        train_epoch(epoch, rank, model, device, data_loader, optimizer)

    end = time.time()
    print(f"Rank {rank} model training time: {end - start:.2f} seconds")


def train_epoch(epoch, rank, model, device, data_loader, optimizer):
    model.train()
    start = time.time()

    for data, target in data_loader:
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = torch.nn.functional.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()

    end = time.time()

    print(
        f"Rank {rank} training epoch {epoch:3d} in {end - start:.2f} seconds: loss = {loss.item():.6f}"
    )
