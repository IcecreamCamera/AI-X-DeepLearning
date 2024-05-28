import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms.functional import resize
from sklearn.manifold import TSNE

from utils import normalize, ResNet50_FPN


def save_loss_graph(train_loss, val_loss=None, path="loss_graph"):
    if not os.path.exists("images"):
        os.mkdir("images")
    
    plt.plot(range(1,len(train_loss)+1), train_loss, label="train")
    if val_loss:
        plt.plot(range(1,len(val_loss)+1), val_loss, label="val")
    plt.title("Loss Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"images/{path}.png")
    plt.clf()


def save_acc_graph(train_acc, val_acc=None, path="accuracy_graph"):
    if not os.path.exists("images"):
        os.mkdir("images")
    
    plt.plot(range(1,len(train_acc)+1), train_acc, label="train")
    if val_acc:
        plt.plot(range(1,len(val_acc)+1), val_acc, label="val")
    plt.title("Acuuracy Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"images/{path}.png")
    plt.clf()


def visualize_filter(model, path="filters"):
    filters = model.conv1.weight.data.cpu()
    filters = np.transpose(np.array(normalize(filters)), (0,2,3,1))
    
    plt.figure(figsize=(16,16))
    plt.suptitle("First Convolutional Layer Filters", fontsize=30)
    for i, filter in enumerate(filters, start=1):
        plt.subplot(8,8,i)
        plt.imshow(filter)
        plt.axis("off")
    plt.savefig(f"images/{path}.png")
    plt.clf()


@torch.no_grad()
def visualize_feature_embedding(model, dataset, path="feaure_embedding"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model_fpn = ResNet50_FPN(model)
    tensor_dataset = TensorDataset(torch.stack(dataset.data), torch.LongTensor(dataset.label))
    dataloader = DataLoader(tensor_dataset, batch_size=128, shuffle=False)

    last_features = np.zeros((len(dataset),2048))
    labels = np.zeros((len(dataset)))
    for i, (data, label) in enumerate(dataloader):
        data = data.to(device)
        last_features[i*128:(i+1)*128] = np.array(model_fpn.extract_last_features(data).cpu())
        labels[i*128:(i+1)*128] = np.array(label)
    
    tsne = TSNE(n_components=2)
    principle_features = tsne.fit_transform(last_features)
    
    plt.figure(figsize=(18,18))
    plt.suptitle("Feature Embedding", fontsize=30)
    for i in np.unique(labels):
        name = dataset.class_name[int(i)]
        if i % 9 == 0:
            plt.subplot(2,2,(int(i)//9)+1)
        plt.scatter(principle_features[labels==i][:,0], principle_features[labels==i][:,1], label=name)
        plt.legend()
    plt.savefig(f"images/{path}.png")
    plt.clf()


@torch.no_grad()
def visualize_cam(model, dataset, path="cam"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    samples = torch.zeros(dataset.num_classes, *dataset.data[0].shape)
    for i in range(dataset.num_classes):
        samples[i] = dataset.data[dataset.label.index(i)]
    samples = samples.to(device)
    
    model_fpn = ResNet50_FPN(model)
    last_feature_map = model_fpn.extract_feature_maps(samples)[-1]
    class_activation_map = torch.matmul(last_feature_map.permute(0,2,3,1), model.fc.weight.data.T).cpu()
    samples = ((samples + 1) / 2).permute(0,2,3,1).cpu()

    plt.figure(figsize=(12,18))
    plt.suptitle("Class Activation Map", fontsize=30)
    for i in range(dataset.num_classes):
        img = np.array(samples[i])
        cam = normalize(resize(class_activation_map[i,:,:,i].unsqueeze(0),
                               img.shape[:-1], torchvision.transforms.InterpolationMode.BICUBIC))
        cam = np.array(cam.permute(1,2,0).repeat(1,1,3))
        plt.subplot(5,7,i+1)
        plt.imshow(np.concatenate([img,cam], axis=0))
        plt.axis("off")
        plt.title(f"{dataset.class_name[i]}")
    plt.savefig(f"images/{path}.png")
    plt.clf()