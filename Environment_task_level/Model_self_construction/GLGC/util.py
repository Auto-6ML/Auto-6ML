import numpy as np
import random
import math
import torch
import torch.nn as nn



L2norm = nn.functional.normalize


def get_mask(view_num, data_size, missing_ratio):
    """
    生成掩码矩阵
    :param view_num: number of views
    :param data_size: size of data
    :param missing_ratio: missing ratio
    :return: mask matrix
    """
    assert view_num >= 2
    miss_sample_num = math.floor(data_size * missing_ratio)
    data_ind = [i for i in range(data_size)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, view_num])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(view_num)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if np.sum(rand_v) > 0 and np.sum(rand_v) < view_num:
                break
        mask[miss_ind[j]] = rand_v
    print('miss_rate:', missing_ratio)
    return mask


def add_gaussian_noise(dataset, mask, noise_std=0.4):
    """
    对指定样本视图特征加高斯噪声
    :param data_X: list of tensors, each tensor shape = [n, d]
    :param mask: tensor of shape [n, v], with 0/1 entries
    :param noise_std: standard deviation of Gaussian noise
    :return: noisy data_X (in-place modified)
    """
    new_feat = []
    n = len(dataset)
    index = torch.arange(n)
    xs, _, _ = dataset.__getitem__(index)
    n, v = mask.shape
    mask = torch.as_tensor(mask)
    for view_idx in range(v):
        # 找出当前视图中需要加噪声的样本索引
        indices = torch.where(mask[:, view_idx] == 0)[0]
        if len(indices) == 0:
            continue
        # 获取当前视图的数据，形状 [n, d]
        x_view = xs[view_idx]
        # 构造高斯噪声
        noise = torch.randn((len(indices), x_view.shape[1]), device=x_view.device) * noise_std
        # 对指定样本加噪声
        x_view[indices] += noise
        new_feat.append(x_view)
    dataset.update_views(new_feat)
    print('node moise finish, std:', noise_std)


@torch.no_grad()
def robust_affinity(hs, mask, args, t=0.07):
    v = len(hs)
    n = hs[0].shape[0]
    G_intra = []
    G_inter = torch.zeros((v, v, n, n), device=hs[0].device)
    hs = [L2norm(hs[h]) for h in range(len(hs))]
    # z2 = [L2norm(z2[i]) for i in range(len(z2))]
    for i in range(v):
        for j in range(v):
            # 计算距离并构造相似度矩阵
            G = (2 - 2 * (hs[i] @ hs[j].t())).clamp(min=0.0)
            G = torch.exp(-G / t)
            if args.choice == 'mask':
                # 获取掩码：哪些样本在视图 i 和 j 中都是存在的
                mask_i = mask[:, i] == 1  # shape: [n]
                mask_j = mask[:, j] == 1

                valid_mask = (mask_i[:, None] & mask_j[None, :])

                # 将无效位置置为 0
                G = G * valid_mask.float()
            if i == j:
                G[torch.eye(n, device=G.device) > 0] = 1.0
            else:
                G[torch.eye(n, device=G.device) > 0] = (
                        G[torch.eye(n, device=G.device) > 0]
                        / G.diag().max().clamp_min(1e-7).detach()
                )

            G = G / G.sum(1, keepdim=True).clamp_min(1e-7)

            if i == j:
                G_intra.append(G)

            G_inter[i, j] = G

        # 融合 G_intra（仍保持为 list）和 G_inter（[v,v,n,n]）
        # 下面这一步是可选的，若你想进一步融合：
    for i in range(v):
        for j in range(v):
            if i == j:
                G_intra[i] = G_intra[i].mm(G_intra[i].t())  # [n, n]
            else:
                G_inter[i, j] = G_inter[i, j].mm(G_inter[j, i].t())
                G_inter[i, j] += args.eta * torch.eye(n, device=G_inter[i, j].device)

    return G_intra, G_inter  # G_intra: list[v][n,n], G_inter_tensor: [v,v,n,n]
