import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, pos, neg, weight, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.pos = pos
        self.neg = neg
        self.weight = weight
        print(self.weight)
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask


    def forward_inter_mask(self, h_i, h_j, mask_flag, W_inter_ij, W_inter_ji):
        """
        GGC module
        h_i, h_j: [n, d] view features
        mask_flag: [n, 2]，1:valid view，0: missing view
        """
        device = h_i.device
        N = 2 * self.batch_size  # N = 2n
        n = self.batch_size
        # Extract diagonal lines as weights for positive sample pairs
        w_pos_i_j = W_inter_ij.diag()  # [n]
        w_pos_j_i = W_inter_ji.diag()  # [n]
        w_pos = torch.cat([w_pos_i_j, w_pos_j_i], dim=0).reshape(N, 1).to(device)  # [2n, 1]
        # concatenate all view features h
        h = torch.cat((h_i, h_j), dim=0)  # [2n, d]

        sim = torch.matmul(h, h.T) / self.temperature_f  # [2n, 2n]

        sim_i_j = torch.diag(sim, n)  # h_i ↔ h_j
        sim_j_i = torch.diag(sim, -n)  # h_j ↔ h_i
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # [2n, 1]

        # weighted
        if self.weight:
            positive_samples = positive_samples * w_pos

        keep_mask = (mask_flag[:, 0] * mask_flag[:, 1]).bool()  # [n]

        valid_mask = torch.cat((keep_mask, keep_mask), dim=0)  # [2n]

        # Negative samples
        mask = self.mask_correlated_samples(N).to(h.device)
        negative_samples = sim[mask].reshape(N, -1)  # [2n, 2n-2]

        logits = torch.cat((positive_samples, negative_samples), dim=1)  # [2n, 1+neg]
        logits = logits[valid_mask]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=h.device)  # [有效样本数]

        loss = self.criterion(logits, labels)
        if valid_mask.sum() == 0:
            loss /= (valid_mask.sum() + 1e-8)
            return loss
        loss /= valid_mask.sum()
        return loss

    def forward_inter_noise(self, h_i, h_j, W_inter_ij, W_inter_ji):
        """
        GGC module
        h_i, h_j: Tensor of shape [n, d]，view feature
        W_inter_ij: Tensor of shape [n, n]，indicate view_i -> view_j Weight of sample pairs
        W_inter_ji: Tensor of shape [n, n]，indicate view_j -> view_i Weight of sample pairs
        """
        device = h_i.device
        n = h_i.size(0)
        N = 2 * n

        # concatenated features
        h = torch.cat([h_i, h_j], dim=0)  # [2n, d]
        sim = torch.matmul(h, h.T) / self.temperature_f  # [2n, 2n]


        # Positive sample similarity
        sim_i_j = torch.diag(sim, n)  # i -> j
        sim_j_i = torch.diag(sim, -n)  # j -> i
        pos_sim = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 1)  # [2n, 1]

        # Extract diagonal lines as weights for positive sample pairs
        w_pos_i_j = W_inter_ij.diag()  # [n]
        w_pos_j_i = W_inter_ji.diag()  # [n]
        w_pos = torch.cat([w_pos_i_j, w_pos_j_i], dim=0).reshape(N, 1).to(device)  # [2n, 1]

        # weighted
        if self.weight:
            pos_sim = pos_sim * w_pos

        # Construct negative samples
        neg_mask = self.mask_correlated_samples(N).to(device)
        neg_sim = sim[neg_mask].reshape(N, -1)  # [2n, 2n-2]

        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [2n, 1+neg]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)

        if logits.size(0) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = self.criterion(logits, labels)
        loss /= logits.size(0)


        return loss



    def forward_intre_mask(self, hs, mask):
        """
        LWC module
        hs: list of V tensors, each [n, d]
        W_intra: tensor of shape [V, n, n]
        W_inter: tensor of shape [V, V, n, n]
        """

        V = len(hs)
        N = hs[0].size(0)

        valid_features = []

        for v in range(V):
            for i in range(N):
                if mask[i][v] == 1:
                    valid_features.append(hs[v][i])
        valid_features = torch.stack(valid_features)

        h = valid_features  # [N, d]
        N = h.size(0)

        # global-view graph
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim.fill_diagonal_(-float('inf'))

        k_pos = max(1, int(self.pos * (N - 1)))
        k_neg = max(1, int(self.neg * (N - 1)))

        # select the top-pos positive samples and the bottom-neg negative samples
        pos_values, pos_indices = torch.topk(sim, k=k_pos, dim=1, largest=True)
        neg_values, neg_indices = torch.topk(sim, k=k_neg, dim=1, largest=False)

        row_idx = torch.arange(N).unsqueeze(1).to(h.device)  # [N, 1]

        pos_sim = sim[row_idx.expand(-1, k_pos), pos_indices]  # [N, k_pos]
        neg_sim = sim[row_idx.expand(-1, k_neg), neg_indices]  # [N, k_neg]

        # [N*k_pos, 1] 和 [N*k_neg, 1]
        pos_sim = pos_sim.reshape(N, -1)
        neg_sim = neg_sim.reshape(N, -1)

        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [Total sample size, 2]
        labels = torch.zeros(N).to(pos_sim.device).long()

        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_intre_noise(self, hs):
        """
        LWC module
        hs: list of V tensors, each [n, d]
        W_intra: tensor of shape [V, n, n]
        W_inter: tensor of shape [V, V, n, n]
        """
        V = len(hs)
        n, d = hs[0].shape

        h = torch.cat(hs, dim=0)  # shape = [V*n, d]

        # global-view graph
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim.fill_diagonal_(-float('inf'))


        k_pos = max(1, int(self.pos * (V * n - 1)))
        k_neg = max(1, int(self.neg * (V * n - 1)))

        # select the top-pos positive samples and the bottom-neg negative samples
        pos_values, pos_indices = torch.topk(sim, k=k_pos, dim=1, largest=True)
        neg_values, neg_indices = torch.topk(sim, k=k_neg, dim=1, largest=False)

        row_idx = torch.arange(V * n).unsqueeze(1).to(h.device)

        pos_sim = sim[row_idx.expand(-1, k_pos), pos_indices]
        neg_sim = sim[row_idx.expand(-1, k_neg), neg_indices]

        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(V * n, dtype=torch.long, device=h.device)

        loss = self.criterion(logits, labels)
        loss /= (V * n)
        return loss



