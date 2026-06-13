import torch.nn as nn
from torch.nn.functional import normalize
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.convs = []
        for v in range(view):
            self.convs.append(HGNN_conv(feature_dim, feature_dim, 10))
        self.convs = nn.ModuleList(self.convs)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            # Varying the number of layers of W can obtain the representations with different shapes.
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view

    def forward(self, xs):
        hs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return hs, xrs, zs


class HConstructor(nn.Module):
    def __init__(self, num_edges, f_dim, iters=1, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_edges = num_edges
        self.edges = None
        self.iters = iters
        self.eps = eps
        self.scale = f_dim ** -0.5

        self.edges_mu = nn.Parameter(torch.randn(1, f_dim))
        self.edges_logsigma = nn.Parameter(torch.zeros(1, f_dim))
        init.xavier_uniform_(self.edges_logsigma)

        self.to_q = nn.Linear(f_dim, f_dim)
        self.to_k = nn.Linear(f_dim, f_dim)
        self.to_v = nn.Linear(f_dim, f_dim)

        self.gru = nn.GRUCell(f_dim, f_dim)

        hidden_dim = max(f_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(f_dim + f_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, f_dim)
        )

        self.norm_input = nn.LayerNorm(f_dim)
        self.norm_edgs = nn.LayerNorm(f_dim)
        self.norm_pre_ff = nn.LayerNorm(f_dim)

    def mask_attn(self, attn, k):
        indices = torch.topk(attn, k).indices
        mask = torch.zeros(attn.shape).bool().to(attn.device)
        for i in range(attn.shape[0]):
            mask[i][indices[i]] = 1
        return attn.mul(mask)

    def ajust_edges(self, s_level):

        if s_level > 0.95:
            self.num_edges = self.num_edges + 1
        elif s_level < 0.9:
            self.num_edges = self.num_edges - 1
            self.num_edges = max(self.num_edges, 4)
        else:
            return

    def forward(self, inputs):
        n, d, device = *inputs.shape, inputs.device
        n_s = self.num_edges

        if True:
            # if self.edges is None:
            mu = self.edges_mu.expand(n_s, -1)
            sigma = self.edges_logsigma.exp().expand(n_s, -1)
            edges = mu + sigma * torch.randn(mu.shape, device=device)
        else:
            edges = self.edges

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        k = F.relu(k)
        v = F.relu(v)

        for _ in range(self.iters):
            edges = self.norm_edgs(edges)

            # 求结点相对于边的softmax
            q = self.to_q(edges)
            q = F.relu(q)

            dots = torch.einsum('ni,ij->nj', q, k.T) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)
            attn = self.mask_attn(attn, 10)  # 这个决定边的特征从哪些结点取

            # 更新超边特征
            updates = torch.einsum('in,nf->if', attn, v)
            edges = torch.cat((edges, updates), dim=1)
            edges = self.mlp(edges)

            # 按边相对于结点的softmax（更新边之后）
            q = self.to_q(inputs)
            k = self.to_k(edges)
            k = F.relu(k)
            q = F.relu(q)  # v = F.relu(v)

            dots = torch.einsum('ni,ij->nj', q, k.T) * self.scale
            attn_v = dots.softmax(dim=1)
            attn_v = self.mask_attn(attn_v, 10)  # 这个决定一个结点属于多少条边
            H = attn_v

            # 计算边的饱和度
            cc = H.ceil().abs()
            de = cc.sum(dim=0)
            empty = (de == 0).sum()
            s_level = 1 - empty / n_s

            self.ajust_edges(s_level)

            print("Num edges is: {}; Satuation level is: {}".format(self.num_edges, s_level))

        self.edges = edges

        return edges, H, dots


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, num_edges, bias=True):
        super(HGNN_conv, self).__init__()

        self.HConstructor = HConstructor(num_edges, in_ft)

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(in_ft, out_ft))
        self.mlp.append(nn.Linear(out_ft, out_ft))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        edges, H, H_raw = self.HConstructor(x)
        edges = edges.matmul(self.weight)
        if self.bias is not None:
            edges = edges + self.bias
        nodes = H.matmul(edges)
        # x = self.mlp[0](x) + self.mlp[1](nodes)
        x = x + nodes
        return x, H, H_raw