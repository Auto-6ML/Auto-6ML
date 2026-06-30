
from dataclasses import dataclass

@dataclass
class Config:
    data_root: str = "./datasets"
    num_classes: int = 10
    num_epochs: int = 600
    warmup_epochs: int = 10
    batch_size: int = 128
    num_workers: int = 4
    noise_rate: float = 0.4
    noise_type: str = "symmetric"  # symmetric 或 asymmetric
    val_ratio: float = 0.1
    clean_ratio: float = 0.1
    noisy_ratio: float = 0.1
    resplit_interval: int = 30
    stop_resplit_epoch: int = 500
    lr_main: float = 0.1
    lr_meta: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lambda_vc: float = 0.05
    nl_weight: float = 1.0
    complex_weight: float = 1.0
    grad_norm_weight: float = 0.5
    grad_cos_weight: float = 0.5
    seed: int = 42
    log_path: str = "logs/train_log.csv"
