def compute_acc(pred, label):
    return (pred.argmax(dim=-1) == label).sum() / len(label)


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


class ResNet50_FPN():
    def __init__(self, resnet50):
        self.resnet50 = resnet50
    
    def extract_feature_maps(self, x):
        out1 = self.resnet50.maxpool(self.resnet50.relu(self.resnet50.bn1(self.resnet50.conv1(x))))
        out2 = self.resnet50.layer1(out1)
        out3 = self.resnet50.layer2(out2)
        out4 = self.resnet50.layer3(out3)
        out5 = self.resnet50.layer4(out4)
        return (out1, out2, out3, out4, out5)
    
    def extract_last_features(self, x):
        return self.extract_feature_maps(x)[-1].mean(dim=(2,3))
