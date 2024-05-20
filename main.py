import argparse
import torch
from torch.utils.data import DataLoader

from model import SimpleCNN, make_resnet50
from dataset import FoodDataset
from train import train
from test import test


DATA_PATH = "FoodClassification"


def parse_args():
    parser = argparse.ArgumentParser(description='Food Image Classification')

    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--pretrained', default=0, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model_type', default="simplecnn", type=str)
    parser.add_argument('--ckpt_path', default=None, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train_dataset = FoodDataset(f"{DATA_PATH}/train")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        model = SimpleCNN(train_dataset.num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        train(model, args, train_dataloader, optimizer, criterion=criterion)

    elif args.mode == "test":
        test_dataset = FoodDataset(f"{DATA_PATH}/test")
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = SimpleCNN(test_dataset.num_classes)
        model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"))

        test(model, test_dataloader)

    elif args.mode == "visualization":
        test_dataset = FoodDataset(f"{DATA_PATH}/test")
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
