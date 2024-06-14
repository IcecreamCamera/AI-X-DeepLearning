import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class FoodDataset(Dataset):
    def __init__(self, path, mode, augment=True):
        self.mode = mode
        self.data = []
        self.label = []
        self.class_name = {}
        self.num_classes = len(os.listdir(f"{path}/{mode}"))

        if mode == "train" and augment:
            self.to_tensor = transforms.Compose([transforms.Resize((256,256)),
                                                transforms.RandAugment(2),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        elif mode == "test":
            self.to_tensor = transforms.Compose([transforms.Resize((256,256)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        
        for i, name in enumerate(os.listdir(f"{path}/{mode}")):
            self.class_name[i] = name
            for img_path in os.listdir(f"{path}/{mode}/{name}"):
                img = Image.open(f"{path}/{mode}/{name}/{img_path}")
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if self.mode == "train":
                    self.data.append(img)
                elif self.mode == "test":
                    self.data.append(self.to_tensor(img))
                self.label.append(i)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == "train":
            data = self.to_tensor(self.data[idx])
        elif self.mode == "test":
            data = self.data[idx]
        label = self.label[idx]
        return data, label