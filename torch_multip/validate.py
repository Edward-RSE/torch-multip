import time

import torch


def validate_model(args, model, device, dataset, dataloader_kwargs):
    start = time.time()
    torch.manual_seed(args.seed)
    data_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            loss += torch.nn.functional.nll_loss(
                output, target.to(device), reduction="sum"
            ).item()
            pred = output.max(1)[1]
            correct += pred.eq(target.to(device)).sum().item()

    loss /= len(data_loader.dataset)
    percent_correct = correct / len(data_loader.dataset) * 100.0
    print(f"Accuracy = {correct} / {len(data_loader.dataset)} ({percent_correct:.0f}%)")
    print(f"Average loss = {loss:.4f}")

    end = time.time()
    print(f"Model validation time: {end - start}")
