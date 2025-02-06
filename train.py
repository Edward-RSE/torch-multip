import os
import torch
import torch.utils.data.dataloader


def train(rank, args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)
    data_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.num_epochs + 1):
        train_epoch(epoch, args, model, device, data_loader, optimizer)


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()

    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = torch.nn.functional.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()

    print(
        "Process {}: training epoch {}: loss {:.6f}".format(
            pid,
            epoch,
            loss.item(),
        )
    )
