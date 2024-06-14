import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import make_resnet50


DATA_PATH = "FoodClassification"
CLASS = [*os.listdir(f"{DATA_PATH}/train")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Food Image Classification')
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--img', default=None, type=str)
    args = parser.parse_args()

    model = make_resnet50(len(CLASS), pretrained=False)
    sd = torch.load(f"checkpoint/{args.ckpt}.pth", map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    to_tensor = transforms.Compose([transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    img = Image.open(args.img).convert("RGB")
    img = to_tensor(img).unsqueeze(0)

    with torch.no_grad():
        logit = model(img)
        idx = logit.argmax(dim=-1).item()
        pred = CLASS[idx]
        prob = torch.softmax(logit, dim=-1)[0,idx].item() * 100
        print(f"This Food is {pred} ({round(prob,2)}%)")
