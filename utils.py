def compute_acc(pred, label):
    return (pred.argmax(dim=-1) == label).sum() / len(label) 