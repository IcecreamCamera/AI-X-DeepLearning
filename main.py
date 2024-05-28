import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import SimpleCNN, make_resnet50
from dataset import FoodDataset
from train import train
from test import test
from visualization import visualize_filter, visualize_feature_embedding, visualize_cam


DATA_PATH = "FoodClassification"


def parse_args():
    parser = argparse.ArgumentParser(description='Food Image Classification')

    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--pretrained', default=1, type=int)
    parser.add_argument('--model_type', default="resnet", type=str)
    parser.add_argument('--model_name', default="model", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    args = parse_args()

    if args.mode == "train":
        train_dataset = FoodDataset(f"{DATA_PATH}/train")
        test_dataset = FoodDataset(f"{DATA_PATH}/test")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        if args.model_type == "simplecnn":
            model = SimpleCNN(train_dataset.num_classes)
        elif args.model_type == "resnet":
            model = make_resnet50(train_dataset.num_classes, pretrained=args.pretrained)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        train(model, args, train_dataloader, optimizer, criterion=criterion, val_dataloader=test_dataloader)

    elif args.mode == "test":
        test_dataset = FoodDataset(f"{DATA_PATH}/test")
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        if args.model_type == "simplecnn":
            model = SimpleCNN(test_dataset.num_classes)
        elif args.model_type == "resnet":
            model = make_resnet50(test_dataset.num_classes, pretrained=False)
        model.load_state_dict(torch.load(f"checkpoint/{args.model_name}.pth", map_location="cpu"))

        test(model, test_dataloader)

    elif args.mode == "visualization":
        test_dataset = FoodDataset(f"{DATA_PATH}/test")
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        if args.model_type == "simplecnn":
            model = SimpleCNN(test_dataset.num_classes)
        elif args.model_type == "resnet":
            model = make_resnet50(test_dataset.num_classes, pretrained=False)
        model.load_state_dict(torch.load(f"checkpoint/{args.model_name}.pth", map_location="cpu"))
        model.eval()
        
        visualize_filter(model)
        visualize_feature_embedding(model, test_dataset)
        visualize_cam(model, test_dataset)
