import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class FoodDataset(Dataset):
    def __init__(self, path):
        self.data = []
        self.label = []
        self.class_name = {}
        self.num_classes = len(os.listdir(path))

        for i, name in enumerate(os.listdir(path)):
            self.class_name[i] = name
            for img_path in os.listdir(f"{path}/{name}"):
                img = Image.open(f"{path}/{name}/{img_path}")
                if img.mode != "RGB":
                    img = img.convert("RGB")
                self.data.append(img)
                self.label.append(i)

        self.transform = transforms.Compose([transforms.Resize((256,256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.transform(self.data[idx])
        label = self.label[idx]
        return data, label