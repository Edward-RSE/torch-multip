import time

import torch


def validate_model(args, model, device, dataloader):
    start = time.time()
    torch.manual_seed(args.seed)

    loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            output = model(data.to(device))
            loss += torch.nn.functional.nll_loss(
                output, target.to(device), reduction="sum"
            ).item()
            pred = output.max(1)[1]
            correct += pred.eq(target.to(device)).sum().item()

    loss /= len(dataloader.dataset)
    percent_correct = correct / len(dataloader.dataset) * 100.0
    print(f"Average loss = {loss:.6f}")
    print(f"Accuracy = {correct} / {len(dataloader.dataset)} ({percent_correct:.0f}%)")

    end = time.time()
    print(f"Model validation time: {end - start:.2f}")
