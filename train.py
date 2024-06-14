import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import compute_acc
from visualization import save_loss_graph, save_acc_graph


def train(model, args, train_dataloader, optim, val_dataloader=None, lr_scheduler=None, criterion=nn.CrossEntropyLoss):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")

    f = open(f"{args.model_name}_log.txt", "w")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    max_val_acc = 0.0
    train_acc, train_loss = [], []
    val_acc, val_loss = [], []

    for epoch in range(args.epoch):
        model.train()
        train_epoch_acc, train_epoch_loss = [], []
        for data, label in tqdm(train_dataloader):
            data, label = data.to(device), label.to(device)
            pred = model(data)
            acc = compute_acc(pred, label)
            loss = criterion(pred, label)
            
            train_epoch_acc.append(acc)
            train_epoch_loss.append(loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

        if lr_scheduler:
            lr_scheduler.step()

        train_epoch_acc = torch.stack(train_epoch_acc).mean().item()
        train_epoch_loss = torch.stack(train_epoch_loss).mean().item()
        print(f"Epoch {epoch+1}/{args.epoch}  train accuracy: {train_epoch_acc}  train loss: {train_epoch_loss}")
        f.write(f"Epoch {epoch+1}/{args.epoch}  train accuracy: {train_epoch_acc}  train loss: {train_epoch_loss}\n")

        train_acc.append(train_epoch_acc)
        train_loss.append(train_epoch_loss)

        if val_dataloader:
            model.eval()
            val_epoch_acc, val_epoch_loss = [], []
            with torch.no_grad():
                for data, label in val_dataloader:
                    data, label = data.to(device), label.to(device)
                    pred = model(data)
                    acc = compute_acc(pred, label)
                    loss = criterion(pred, label)

                    val_epoch_acc.append(acc)
                    val_epoch_loss.append(loss)
            
            val_epoch_acc = torch.stack(val_epoch_acc).mean().item()
            val_epoch_loss = torch.stack(val_epoch_loss).mean().item()
            print(f"Epoch {epoch+1}/{args.epoch}  val accuracy: {val_epoch_acc}  val loss: {val_epoch_loss}")
            f.write(f"Epoch {epoch+1}/{args.epoch}  val accuracy: {val_epoch_acc}  val loss: {val_epoch_loss}\n")

            val_acc.append(val_epoch_acc)
            val_loss.append(val_epoch_loss)
                
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint/{args.model_name}{epoch+1}.pth")
        
        if val_epoch_acc > max_val_acc:
            torch.save(model.state_dict(), f"checkpoint/{args.model_name}_maxval.pth")
            max_val_acc = val_epoch_acc

    save_acc_graph(train_acc, val_acc, f"{args.model_name}_acc")
    save_loss_graph(train_loss, val_loss, f"{args.model_name}_loss")
    