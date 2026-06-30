
import numpy as np


def apply_symmetric_noise(labels, noise_rate, num_classes=10, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).copy()
    clean = labels.copy()
    n_noisy = int(len(labels) * noise_rate)
    noisy_idx = rng.choice(len(labels), n_noisy, replace=False)
    for idx in noisy_idx:
        candidates = list(range(num_classes))
        candidates.remove(int(labels[idx]))
        labels[idx] = rng.choice(candidates)
    return labels.tolist(), clean.tolist(), noisy_idx.tolist()


def apply_asymmetric_noise(labels, noise_rate, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).copy()
    clean = labels.copy()
    # CIFAR-10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    mapping = {9: 1, 2: 0, 4: 7, 3: 5, 5: 3}
    candidates = [i for i, y in enumerate(labels) if int(y) in mapping]
    n_noisy = int(len(labels) * noise_rate)
    noisy_idx = rng.choice(candidates, min(n_noisy, len(candidates)), replace=False)
    for idx in noisy_idx:
        labels[idx] = mapping[int(labels[idx])]
    return labels.tolist(), clean.tolist(), noisy_idx.tolist()
