import torch
import torch.nn.functional as F


def reconstruction_loss(x_full, x_gen, mask, eps=1e-8):
        return (((x_full - x_gen) ** 2) * mask).sum() / (mask.sum() + eps)


def kl_loss(mu, logvar):
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def discriminator_loss(d_prob, mask, eps=1e-8):
        return F.binary_cross_entropy(d_prob.clamp(eps, 1 - eps), mask)


def generator_adversarial_loss(d_prob, mask, eps=1e-8):
       missing = 1.0 - mask
    target = torch.ones_like(d_prob)
    bce = F.binary_cross_entropy(d_prob.clamp(eps, 1 - eps), target, reduction="none")
    return (bce * missing).sum() / (missing.sum() + eps)


def classification_loss(classifier, x_hat, y):
        return F.cross_entropy(classifier(x_hat), y)


def armse(x_true, x_pred, mask, eps=1e-8, strict_formula=True):
    missing = 1.0 - mask
    d = x_true.shape[1]
    values = []
    total = x_true.new_tensor(0.0)
    for j in range(d):
        miss_j = missing[:, j]
        denom = miss_j.sum()
        if denom > 0:
            rmse_j = torch.sqrt((((x_true[:, j] - x_pred[:, j]) ** 2) * miss_j).sum() / (denom + eps))
            total = total + rmse_j
            values.append(rmse_j)
    if strict_formula:
        return total / d
    if len(values) == 0:
        return x_true.new_tensor(0.0)
    return torch.stack(values).mean()


def make_hint(mask, hint_rate):
       hint_selector = (torch.rand_like(mask) < hint_rate).float()
    return hint_selector * mask + (1.0 - hint_selector) * 0.5


def loss_stats_tensor(rec, kl, adv, cls, epoch_ratio):
       stats = torch.stack([
        rec.detach(), kl.detach(), adv.detach(), cls.detach(),
        torch.as_tensor(epoch_ratio, device=rec.device, dtype=rec.dtype),
    ])
    stats[:4] = torch.log1p(torch.clamp(stats[:4], min=0.0))
    return stats.view(1, -1)
