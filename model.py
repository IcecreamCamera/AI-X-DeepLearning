import torch.nn as nn
import torchvision


def make_resnet50(num_classes, pretrained=True):
    model = torchvision.models.resnet50(weights="DEFAULT" if pretrained else None)
    model.fc = nn.Linear(2048,num_classes)
    return model


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.ModuleList([self.make_block(3,32),
                                  self.make_block(32,64),
                                  self.make_block(64,128),
                                  self.make_block(128,256)])
        self.fc = nn.Linear(256,num_classes)

    def make_block(self, in_c, out_c):
        modules = [nn.Conv2d(in_c, out_c, 3, padding=1),
                   nn.BatchNorm2d(out_c),
                   nn.ReLU(),
                   nn.MaxPool2d(2)]
        return nn.Sequential(*modules)
    
    def forward(self, x):
        for module in self.cnn:
            x = module(x)
        x = x.mean(dim=(2,3))
        x = self.fc(x)
        return x