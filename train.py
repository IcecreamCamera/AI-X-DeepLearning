import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import compute_acc


def train(model, args, dataloader, optim, lr_scheduler=None, criterion=nn.CrossEntropyLoss):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(args.epoch):
        epoch_acc, epoch_loss = [], []

        for data, label in tqdm(dataloader):
            data, label = data.to(device), label.to(device)
            
            pred = model(data)
            acc = compute_acc(pred, label)
            loss = criterion(pred, label)
            
            epoch_acc.append(acc)
            epoch_loss.append(loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

        if lr_scheduler:
            lr_scheduler.step()

        epoch_acc = torch.stack(epoch_acc).mean().item()
        epoch_loss = torch.stack(epoch_loss).mean().item()
        print(f"Epoch {epoch}/{args.epoch}  accuracy: {epoch_acc}, loss: {epoch_loss}\n")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint/model{epoch+1}.pth")