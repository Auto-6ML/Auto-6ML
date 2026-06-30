
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from config import Config
from data.augmentations import weak_transform, test_transform
from data.cifar10_noisy import build_train_val_sets
from models.main_model import MainNet
from models.meta_net import MetaNet
from trainer.sample_splitter import SampleStats
from trainer.meta_trainer import MetaTrainer
from utils.logger import CSVLogger
from utils.seed import set_seed


def main():
  
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type != "cuda":
        print("警告：当前未检测到 GPU，代码仍可运行但速度会明显变慢。")

    
    full_train, train_subset, val_subset = build_train_val_sets(cfg, weak_transform())
    test_set = CIFAR10(root=cfg.data_root, train=False, download=True, transform=test_transform())
   
    class TestWrapper(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            return x, y, y, i
    test_loader = DataLoader(TestWrapper(test_set), batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    main_model = MainNet(num_classes=cfg.num_classes, feature_dim=64).to(device)
    meta_net = MetaNet(feature_dim=64, hidden_dim=200, num_classes=cfg.num_classes).to(device)

    opt_main = torch.optim.SGD(main_model.parameters(), lr=cfg.lr_main,
                               momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    opt_meta = torch.optim.SGD(meta_net.parameters(), lr=cfg.lr_meta,
                               momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    stats = SampleStats(len(full_train), device="cpu")
    logger = CSVLogger(cfg.log_path)
    trainer = MetaTrainer(cfg, full_train, train_subset, val_subset, test_loader,
                          main_model, meta_net, opt_main, opt_meta, stats, logger, device)
    trainer.fit()


if __name__ == "__main__":
    main()
