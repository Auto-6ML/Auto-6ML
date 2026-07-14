
import torch

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval(); correct = total = 0
    for images, _, clean_labels, _ in loader:
        images, labels = images.to(device), clean_labels.to(device)
        pred = model(images).argmax(1)
        correct += (pred == labels).sum().item(); total += labels.numel()
    return 100.0 * correct / max(total, 1)
