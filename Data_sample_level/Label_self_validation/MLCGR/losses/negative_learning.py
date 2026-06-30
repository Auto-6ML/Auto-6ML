
import torch
import torch.nn.functional as F


def negative_learning_loss(logits, forbidden_labels, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    forbidden_prob = probs.gather(1, forbidden_labels.view(-1, 1)).squeeze(1)
    return -torch.log(torch.clamp(1.0 - forbidden_prob, min=eps)).mean()
