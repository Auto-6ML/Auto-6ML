
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.augmentations import weak_transform, strong_transform, test_transform
from data.cifar10_noisy import subset_with_transform
from losses.negative_learning import negative_learning_loss
from losses.feature_constraint import compute_L_vc
from utils.metrics import accuracy


def soft_ce(logits, soft_targets):
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


class MetaTrainer:
    def __init__(self, cfg, full_train, train_subset, val_subset, test_loader,
                 main_model, meta_net, opt_main, opt_meta, stats, logger, device):
        self.cfg, self.full_train = cfg, full_train
        self.train_indices = list(train_subset.indices)
        self.val_subset, self.test_loader = val_subset, test_loader
        self.main_model, self.meta_net = main_model, meta_net
        self.opt_main, self.opt_meta = opt_main, opt_meta
        self.stats, self.logger, self.device = stats, logger, device
        self.clean_idx, self.noisy_idx, self.complex_idx = [], [], self.train_indices

    def _loader(self, indices, transform, shuffle=True):
        ds = subset_with_transform(self.full_train, indices, transform)
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle,
                          num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)

    def _val_loader(self):
        self.val_subset.dataset.transform = strong_transform()
        return DataLoader(self.val_subset, batch_size=self.cfg.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)

    def warmup_epoch(self):
        loader = self._loader(self.train_indices, weak_transform())
        self.main_model.train(); total = 0.0
        for x, y, _, idx in loader:
            x, y, idx = x.to(self.device), y.to(self.device), idx.to(self.device)
            logits, feats = self.main_model(x, return_features=True)
            loss = F.cross_entropy(logits, y) + self.cfg.lambda_vc * compute_L_vc(feats, y, self.cfg.num_classes)
            self.opt_main.zero_grad(); loss.backward(); self.opt_main.step()
            self.stats.update(idx, feats.detach(), logits.detach(), y.detach())
            total += loss.item() * x.size(0)
        return total / len(loader.dataset)

    def resplit(self):
        self.clean_idx, self.noisy_idx, self.complex_idx = self.stats.split(
            self.train_indices, self.cfg.clean_ratio, self.cfg.noisy_ratio,
            self.cfg.grad_norm_weight, self.cfg.grad_cos_weight)

    def train_epoch(self):
        complex_loader = self._loader(self.complex_idx, weak_transform())
        meta_indices = self.clean_idx + self.noisy_idx
        meta_loader = self._loader(meta_indices, strong_transform())
        val_loader = self._val_loader()
        meta_iter, val_iter = iter(meta_loader), iter(val_loader)
        self.main_model.train(); self.meta_net.train(); total = 0.0
        for x, y, _, idx in complex_loader:
            x, y = x.to(self.device), y.to(self.device)
           
            feats = self.main_model.extract_features(x)
            with torch.no_grad():
                pseudo = F.softmax(self.meta_net(feats.detach()), dim=1)
            logits = self.main_model.classifier(feats)
            loss_main = soft_ce(logits, pseudo)
            self.opt_main.zero_grad(); loss_main.backward(); self.opt_main.step()

           
            try:
                mx, my, _, midx = next(meta_iter)
            except StopIteration:
                meta_iter = iter(meta_loader); mx, my, _, midx = next(meta_iter)
            mx, my, midx = mx.to(self.device), my.to(self.device), midx.to(self.device)
            with torch.no_grad():
                mfeat = self.main_model.extract_features(mx)
            meta_logits = self.meta_net(mfeat)
            clean_mask = torch.tensor([int(i) in set(self.clean_idx) for i in midx.cpu().tolist()], device=self.device)
            meta_loss = meta_logits.new_tensor(0.0)
            if clean_mask.any():
                meta_loss = meta_loss + F.cross_entropy(meta_logits[clean_mask], my[clean_mask])
            if (~clean_mask).any():
                pseudo_bad = meta_logits[~clean_mask].argmax(1)
                meta_loss = meta_loss + self.cfg.nl_weight * negative_learning_loss(meta_logits[~clean_mask], pseudo_bad)
            
            try:
                vx, _, vy, _ = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader); vx, _, vy, _ = next(val_iter)
            vx, vy = vx.to(self.device), vy.to(self.device)
            with torch.no_grad():
                vf = self.main_model.extract_features(vx)
            meta_loss = meta_loss + F.cross_entropy(self.meta_net(vf), vy)
            self.opt_meta.zero_grad(); meta_loss.backward(); self.opt_meta.step()

           
            logits, feats = self.main_model(x, return_features=True)
            with torch.no_grad():
                pseudo = F.softmax(self.meta_net(feats.detach()), dim=1)
            loss = self.cfg.complex_weight * soft_ce(logits, pseudo) + self.cfg.lambda_vc * compute_L_vc(feats, y, self.cfg.num_classes)
            self.opt_main.zero_grad(); loss.backward(); self.opt_main.step()
            total += loss.item() * x.size(0)
        return total / max(len(complex_loader.dataset), 1)

    def fit(self):
        for epoch in range(1, self.cfg.num_epochs + 1):
            if epoch <= self.cfg.warmup_epochs:
                train_loss = self.warmup_epoch()
                if epoch == self.cfg.warmup_epochs:
                    self.resplit()
            else:
                if epoch <= self.cfg.stop_resplit_epoch and epoch % self.cfg.resplit_interval == 0:
                    self.resplit()
                train_loss = self.train_epoch()
            test_acc = accuracy(self.main_model, self.test_loader, self.device)
            self.logger.log([epoch, train_loss, test_acc, len(self.clean_idx), len(self.noisy_idx), len(self.complex_idx)])
            print(f"Epoch [{epoch}/{self.cfg.num_epochs}] Loss: {train_loss:.4f} Test Acc: {test_acc:.2f}% "
                  f"Clean/Noisy/Complex={len(self.clean_idx)}/{len(self.noisy_idx)}/{len(self.complex_idx)}")
