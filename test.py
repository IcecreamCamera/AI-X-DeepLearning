import torch

from utils import compute_acc


@torch.no_grad()
def test(model, dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    acc_list = []
    model.eval()
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)

        pred = model(data)
        acc = compute_acc(pred, label)
        acc_list.append(acc)
    
    test_acc = torch.stack(acc_list).mean().item()
    print(f"Test accuracy: {test_acc}")