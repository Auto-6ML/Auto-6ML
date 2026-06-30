
import numpy as np
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import CIFAR10
from .noise import apply_symmetric_noise, apply_asymmetric_noise


class IndexedNoisyCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=True,
                 noise_rate=0.0, noise_type="symmetric", seed=42):
        self.base = CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.data = self.base.data
        self.targets = list(self.base.targets)
        if train and noise_rate > 0:
            if noise_type == "symmetric":
                self.noisy_targets, self.clean_targets, self.noisy_indices = apply_symmetric_noise(
                    self.targets, noise_rate, 10, seed)
            elif noise_type == "asymmetric":
                self.noisy_targets, self.clean_targets, self.noisy_indices = apply_asymmetric_noise(
                    self.targets, noise_rate, seed)
            else:
                raise ValueError("noise_type 必须是 symmetric 或 asymmetric")
        else:
            self.noisy_targets = list(self.targets)
            self.clean_targets = list(self.targets)
            self.noisy_indices = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.base.data[index]
        image = self.base.transforms(image) if False else image
        from PIL import Image
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.noisy_targets[index]), int(self.clean_targets[index]), int(index)


def build_train_val_sets(cfg, transform):
    full = IndexedNoisyCIFAR10(cfg.data_root, train=True, transform=transform, download=True,
                               noise_rate=cfg.noise_rate, noise_type=cfg.noise_type, seed=cfg.seed)
    n_val = int(len(full) * cfg.val_ratio)
    n_train = len(full) - n_val
    gen = __import__('torch').Generator().manual_seed(cfg.seed)
    train_subset, val_subset = random_split(full, [n_train, n_val], generator=gen)
    return full, train_subset, val_subset


def subset_with_transform(full_dataset, indices, transform):
    ds = IndexedNoisyCIFAR10(full_dataset.base.root, train=True, transform=transform, download=False)
    ds.data = full_dataset.data
    ds.targets = full_dataset.targets
    ds.noisy_targets = full_dataset.noisy_targets
    ds.clean_targets = full_dataset.clean_targets
    ds.noisy_indices = full_dataset.noisy_indices
    return Subset(ds, list(map(int, indices)))
