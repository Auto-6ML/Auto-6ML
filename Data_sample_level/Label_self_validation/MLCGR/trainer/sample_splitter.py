
import torch


class SampleStats:
    def __init__(self, dataset_size, device="cpu"):
        self.grad_norms = [[] for _ in range(dataset_size)]
        self.grad_dirs = [[] for _ in range(dataset_size)]
        self.device = device

    @torch.no_grad()
    def update(self, indices, features, logits, labels):
        probs = torch.softmax(logits, dim=1)
        onehot = torch.zeros_like(probs).scatter_(1, labels.view(-1, 1), 1.0)
        grad_logits = probs - onehot
        
        norms = grad_logits.norm(dim=1).detach().cpu()
        dirs = torch.nn.functional.normalize(grad_logits.detach().cpu(), dim=1)
        for j, idx in enumerate(indices.detach().cpu().tolist()):
            self.grad_norms[idx].append(float(norms[j]))
            self.grad_dirs[idx].append(dirs[j])

    def split(self, candidate_indices, clean_ratio=0.1, noisy_ratio=0.1,
              norm_weight=0.5, cos_weight=0.5):
        scores = []
        for idx in candidate_indices:
            norms = self.grad_norms[idx]
            dirs = self.grad_dirs[idx]
            mean_norm = sum(norms) / max(len(norms), 1) if norms else 1.0
            cos_change = 1.0
            if len(dirs) >= 2:
                cos_vals = [torch.dot(dirs[k], dirs[k-1]).item() for k in range(1, len(dirs))]
                cos_change = 1.0 - (sum(cos_vals) / len(cos_vals))
            scores.append(norm_weight * mean_norm + cos_weight * cos_change)
        scores = torch.tensor(scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        order = torch.argsort(scores)  # 低分更干净
        n = len(candidate_indices)
        n_clean = max(1, int(n * clean_ratio))
        n_noisy = max(1, int(n * noisy_ratio))
        cand = torch.tensor(candidate_indices)
        clean = cand[order[:n_clean]].tolist()
        noisy = cand[order[-n_noisy:]].tolist()
        complex_ = cand[order[n_clean:n-n_noisy]].tolist()
        return clean, noisy, complex_
