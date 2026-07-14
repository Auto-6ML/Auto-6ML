
import torch
import torch.nn.functional as F


def compute_g1(features, labels, num_classes=10, eps=1e-6):
    value, count = features.new_tensor(0.0), 0
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() < 2:
            continue
        fc = features[mask]
        center = fc.mean(dim=0, keepdim=True)
        value = value + ((fc - center) ** 2).sum(dim=1).mean()
        count += 1
    return value / max(count, 1) + eps


def compute_g2(features, labels, num_classes=10, eps=1e-6):
    value, count = features.new_tensor(0.0), 0
    for c in range(num_classes):
        pos = labels == c
        neg = labels != c
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        pos_center = features[pos].mean(dim=0)
        neg_center = features[neg].mean(dim=0)
        value = value + torch.sum((pos_center - neg_center) ** 2)
        count += 1
    return value / max(count, 1) + eps


def compute_L_vc(features, labels, num_classes=10):
    features = F.normalize(features, dim=1)
    g1 = compute_g1(features, labels, num_classes)
    g2 = compute_g2(features, labels, num_classes)
    return g1 / g2
