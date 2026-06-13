import os
import torch.nn as nn
from tqdm import tqdm
from evaluate import evaluate
from embedder import embedder
import numpy as np
import random as random
import torch
import copy
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
from collections import OrderedDict
from torch.autograd import Variable
import torch
import torch_cluster.knn as torch_knn
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU

# class Edge_weightedMLP(nn.Module):
#     def __init__(self, args):
#         self.args = args
#         cfg = self.args.cfg
#         super(Edge_weightedMLP, self).__init__()
#         self.mlp_edge_model = Sequential(
#             Linear(cfg[-1] * 4, 64),
#             ReLU(),
#             Linear(64, 1))

class MetricCalcLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        return h * self.weight

class WeightedMLP(nn.Module):
    def __init__(self, args):
        self.args = args
        cfg = self.args.cfg
        super(WeightedMLP, self).__init__()
        view_num = self.args.view_num
        self.w_list = nn.ModuleList([nn.Linear(cfg[-1] *2, cfg[-1], bias=True) for _ in range(view_num)])
        self.y_list = nn.ModuleList([nn.Linear(cfg[-1], 1) for _ in range(view_num)])
        # self.map = nn.ModuleList([nn.Linear(cfg[-1] * 2, cfg[-1] * 2) for _ in range(view_num)])
        # self.W = nn.Parameter(torch.zeros(size=(view_num * cfg[-1], cfg[-1])))
        self.att_act1 = nn.Tanh()
        self.threshold = 0.2
        self.att_act2 = nn.Softmax(dim=-1)
        self.mlp_edge_model = Sequential(
            Linear(cfg[-1] * 4, 128),
            ReLU(),
            Linear(128, 1))
        # s = torch.zeros((self.args.nb_nodes, self.args.nb_nodes)).to(self.args.device)
        # attention = self.features @ self.features.T
        # for i in range(attention.shape[0]):
        #     attention[i][i] = 1
        # kthvalue = torch.kthvalue(attention.view(attention.shape[0] * attention.shape[1], 1).T,
        #                           int(attention.shape[0] * attention.shape[1] * self.args.edge_rate))[0]
        # # torch.use_deterministic_algorithms(True)
        # mask = (attention > kthvalue).detach().float()
        # self.attention = (attention * mask).detach()
        # self.metric_layer = nn.ModuleList(nn.init.xavier_uniform_(self.weight))
        # self.metric_layer = nn.ModuleList()
        # # for i in range(num_head):
        # self.metric_layer.append(MetricCalcLayer(cfg[-1] * 2))

        # self.weight = nn.Parameter(torch.FloatTensor(1, cfg[-1] * 2))
        # nn.init.xavier_uniform_(self.weight)
        # for i in range(2):
        #     self.metric_layer.append(MetricCalcLayer(cfg[-1] * 2))
        # self.deg_enc_list = []
        # for i in range(view_num):
        #     self.deg_enc_list.append(nn.Embedding(int(max(self.args.degree_list[i]) + 1), cfg[-1], padding_idx=0))
        # self.Linear = nn.ModuleList([nn.Linear(cfg[-1], cfg[-1], bias=True)])
        # self.Linear = Sequential(
        #     Linear(cfg[-1], cfg[-1]))

################################################################
    def forward(self, h_list, adj_list, h_a_list, knn):
        h_combine_list = []
        graph_score_list = []
        # edge_logits_list = []
        # meta_adj_list = []

        for i, h in enumerate(h_list):
            # deg_encoding = self.deg_enc_list[i](torch.LongTensor(self.args.degree_list[i])).to(self.args.device)
            # feature_encoding = self.Linear(h_list[i])
            # deg_enc_list.append(deg_encoding)
            # h_cat = torch.cat([h,deg_encoding],dim=1)
            ####################################################
            # s = torch.zeros((h.shape[0], h.shape[0])).to(self.args.device)
            # for i in range(1):
            #     # weighted_left_h = h * self.weight
            #     # weighted_right_h = h * self.weight
            #     weighted_left_h = self.metric_layer[i](h)
            #     # weighted_right_h = self.metric_layer[i](h)
            #     s += cos_sim(weighted_left_h, weighted_left_h)
            # s /= 1
            # S_new = torch.where(s < self.threshold, torch.zeros_like(s), s)
            # # S_new = s
            ###########################################################


            # h_a_map = self.map[i](h_a_list[i])
            # S_new = cosine_dist(h_a_map, h_a_map)
            # # S_new = torch.where(S_new < self.threshold, torch.zeros_like(S_new), S_new)
            # S_new = torch.mul(S_new, adj_list[i])
            # h_new = torch.mm(S_new, h_a_list[i])

            # h = h + h_new
            # h = torch.cat([h, h_new], dim=1)

            g_emb = torch.mean(h, 0)
            g_emb = self.w_list[i](g_emb)
            g_so = self.y_list[i](g_emb)
            graph_score_list.append(g_so)
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
            ###########################
            # src, dst = adj_list[i].to_sparse()._indices()[0], adj_list[i].to_sparse()._indices()[1]
            # # src, dst = edge_index[0], edge_index[1]
            # emb_src = h_cat[src]
            # emb_dst = h_cat[dst]
            # edge_emb = torch.cat([emb_src, emb_dst], 1)
            # edge_logits = self.mlp_edge_model(edge_emb)
            # bias = 0.0001
            # eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            # edge_score = torch.log(eps) - torch.log(1 - eps)
            # edge_score = edge_score.to(self.args.device)
            # edge_score = (edge_score + edge_logits)
            # edge_weight = torch.sigmoid(edge_score).squeeze()
            # adj_list[i][adj_list[i].nonzero().T[0], adj_list[i].nonzero().T[1]] = edge_weight
            # meta_adj_list.append(adj_list[i])


            # edge_logits_list.append(edge_logits)

        g_score = torch.cat(graph_score_list, -1)
        g_score = self.att_act1(g_score)
        g_score = self.att_act2(g_score)
        g_score = torch.unsqueeze(g_score, -1)
        g_score = g_score.unsqueeze(0)

        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = g_score * score * h
        h = torch.sum(h, dim=1)
        ##############################

        feature_embedding = sum(h_a_list) / self.args.view_num
        s = torch.zeros((h.shape[0], h.shape[0])).to(self.args.device)
        src, dst = knn
        emb_src = feature_embedding[src]
        emb_dst = feature_embedding[dst]
        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)
        bias = 0.0001
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        edge_score = torch.log(eps) - torch.log(1 - eps)
        edge_score = edge_score.to(self.args.device)
        edge_score = (edge_score + edge_logits)
        edge_weight = torch.sigmoid(edge_score).squeeze()
        s[src, dst] = edge_weight
        #
        h_new = torch.mm(s, feature_embedding)
        #
        # # h = h + h_new
        h = torch.cat([h, h_new], dim=1)
        # meta_adj_list.append(adj_list[i])

        # meta_adj = sum(adj_list) / self.args.view_num
        # src, dst = meta_adj.to_sparse()._indices()[0], meta_adj.to_sparse()._indices()[1]
        # # src, dst = edge_index[0], edge_index[1]
        # emb_src = h_cat[src]
        # emb_dst = h_cat[dst]
        # edge_emb = torch.cat([emb_src, emb_dst], 1)
        # edge_logits = self.mlp_edge_model(edge_emb)
        # bias = 0.0001
        # eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        # edge_score = torch.log(eps) - torch.log(1 - eps)
        # edge_score = edge_score.to(self.args.device)
        # edge_score = (edge_score + edge_logits)
        # edge_weight = torch.sigmoid(edge_score).squeeze()
        # meta_adj[meta_adj.nonzero().T[0], meta_adj.nonzero().T[1]] = edge_weight
        # meta_adj_list.append(adj_list[i])
        #################################
        # S_new = cosine_dist(h, h)
        # S_new = torch.mul(S_new, meta_adj)


        # meta_adj = sum(meta_adj_list) / self.args.view_num
        # h = torch.mm(S_new, h)
        # print(score)
        # h_list = h_list
##################edge weight####################################
        # for i, adj in enumerate(adj_list):
        #     src, dst= adj.to_sparse()._indices()[0], adj.to_sparse()._indices()[0]
        #     # src, dst = edge_index[0], edge_index[1]
        #     emb_src = node_emb[src]
        #     emb_dst = node_emb[dst]



        return h, score, g_score, s
        #############################################

    def embed(self, h_list, adj_list, h_a_list, knn):
        h_combine_list = []
        for i, h in enumerate(h_list):
            # deg_encoding = self.deg_enc_list[i](torch.LongTensor(self.args.degree_list[i])).to(self.args.device)
            # deg_enc_list.append(deg_encoding)
            # h = torch.cat([h, deg_encoding], dim=1)
            # h_a_map = self.map[i](h_a_list[i])
            # S_new = cosine_dist(h_a_map, h_a_map)
            # # S_new = torch.where(S_new < self.threshold, torch.zeros_like(S_new), S_new)
            # S_new = torch.mul(S_new, adj_list[i])
            # h_new = torch.mm(S_new, h_a_list[i])
            #
            # # h = h + h_new
            # h = torch.cat([h, h_new], dim=1)

            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)

        # s = torch.zeros((h.shape[0], h.shape[0])).to(self.args.device)
        feature_embedding = sum(h_a_list) / self.args.view_num
        s = torch.zeros((h.shape[0], h.shape[0])).to(self.args.device)
        src, dst = knn
        emb_src = feature_embedding[src]
        emb_dst = feature_embedding[dst]
        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)
        bias = 0.0001
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        edge_score = torch.log(eps) - torch.log(1 - eps)
        edge_score = edge_score.to(self.args.device)
        edge_score = (edge_score + edge_logits)
        edge_weight = torch.sigmoid(edge_score).squeeze()
        s[src, dst] = edge_weight

        h_new = torch.mm(s, feature_embedding)

        # h = h + h_new
        h = torch.cat([h, h_new], dim=1)

        return h.detach()


def cosine_dist(x, y):
    bs1 = x.size()[0]
    bs2 = y.size()[0]
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2) + 0.00000001, 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2) + 0.00000001, 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return cosine

class SHG_v2(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.criteria = nn.BCEWithLogitsLoss()
        self.cfg = args.cfg
        self.sigm = nn.Sigmoid()
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)
        self.model = trainer(self.args)
        self.model = self.model.to(self.args.device)
        self.model_copy = copy.deepcopy(self.model)


        self.meta_model = WeightedMLP(self.args)
        self.meta_model = self.meta_model.to(self.args.device)


    def training(self):
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # # ===================================================#
        features = self.features#.to(self.args.device)
        adj_list = self.adj_list#[adj.to(self.args.device) for adj in self.adj_list]
        print("Started training...")

        # self.knn_edge = self.knn_edge.to(self.args.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        optimiser_meta = torch.optim.Adam(self.meta_model.parameters(), lr=self.args.meta_lr)
        # s = torch.zeros((features.shape[0], features.shape[0])).to(self.args.device)
        # knn_graph = knn.get_knns(features, k=5)

        # attention = features @ features.T
        # for i in range(attention.shape[0]):
        #     attention[i][i] = 1
        # kthvalue = torch.kthvalue(attention.view(attention.shape[0] * attention.shape[1], 1).T,
        #                           int(attention.shape[0] * attention.shape[1] * self.args.edge_rate))[0]
        # # torch.use_deterministic_algorithms(True)
        # mask = (attention > kthvalue).detach().float()
        # attention = (attention * mask)
        # # s = torch.zeros((features.shape[0], features.shape[0])).to(self.args.device)
        # # knn_edge = []
        # knn_edge = attention.to_sparse()._indices()[0], attention.to_sparse()._indices()[1]
        # attention.remove_edges_from(attention.selfloop_edges(attention))

        # self.model.train()

        for epoch in tqdm(range(self.args.nb_epochs+1)):
    ###################################train model########################################
            self.model.train()
            self.meta_model.eval()
            optimiser.zero_grad()

            x_list, h_p_list, h_a_list = self.model(features, adj_list, self.idx_p_list, epoch) #, recons
            fusion_emb, node_weight, g_score, edge_weight = self.meta_model(h_p_list, adj_list, h_a_list, self.knn_edge)

            # s[knn_edge[0], knn_edge[1]] = edge_weight
            # h_new = torch.mm(edge_weight, h_a_list[0])
            # fusion_emb = torch.cat([fusion_emb, h_new], dim=1)

            # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)

            loss_inv_view, loss_intra_view = View_intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
            loss_inv, loss_intra = intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
            loss = loss_inv + self.args.inv_view_weight * loss_inv_view + self.args.alpha * loss_intra_view #loss_intra #+ self.args.beta * (recons_err + recons_nei)


            loss.backward()
            optimiser.step()

##############################################train meta##############################################
            self.model.eval()
            self.meta_model.train()
            optimiser_meta.zero_grad()

            x_list, h_p_list, h_a_list = self.model(features, adj_list, self.idx_p_list, epoch) #, recons
            fusion_emb, node_weight, g_score, edge_weight = self.meta_model(h_p_list, adj_list, h_a_list, self.knn_edge)

            # s[knn_edge[0], knn_edge[1]] = edge_weight
            # h_new = torch.mm(edge_weight, h_a_list[0])
            # fusion_emb = torch.cat([fusion_emb, h_new], dim=1)

            # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
            loss_inv_view, loss_intra_view = View_intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
            loss_inv, loss_intra = intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
            loss = loss_inv + self.args.inv_view_weight * loss_inv_view + self.args.alpha * loss_intra_view #loss_intra #+ self.args.beta * (recons_err + recons_nei)

            # current parameter
            # fast_weights = OrderedDict((name, param) for (name, param) in self.model.named_parameters())

            # create_graph flag for computing second-derivative
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # data = [p.data for p in list(self.model.parameters())]

            with torch.no_grad():
                for weight, weight_, grad in zip(self.model.parameters(), self.model_copy.parameters(),
                                                 grads):
                    if 'momentum' in optimiser.param_groups[0].keys():  # used in SGD with momentum
                        if optimiser.param_groups[0]['momentum'] == 0:
                            m = 0
                        else:
                            m = optimiser.state[weight].get('momentum_buffer', 0.) * optimiser.param_groups[0]['momentum']
                    else:
                        m = 0
                    weight_.copy_(weight - 0.1 * (m + grad + optimiser.param_groups[0]['weight_decay'] * weight))



            # compute parameter' by applying sgd on multi-task loss
            # fast_weights = OrderedDict(
            #     (name, param - 0.1 * grad) for ((name, param), grad, data)  #args.LGA_lr
            #     in zip(fast_weights.items(), grads, data))  # \fi0 and \varfi0

            # compute primary loss with the updated parameter'

            x_list, h_p_list, h_a_list = self.model_copy(features, adj_list, self.idx_p_list, epoch) #, recons
            fusion_emb, node_weight, g_score, edge_weight = self.meta_model(h_p_list, adj_list, h_a_list, self.knn_edge)

            # s[knn_edge[0], knn_edge[1]] = edge_weight
            # h_new = torch.mm(edge_weight, h_a_list[0])
            # fusion_emb = torch.cat([fusion_emb, h_new], dim=1)

            # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
            loss_inv_view, loss_intra_view = View_intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
            loss_inv, loss_intra = intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
            loss = loss_inv + self.args.inv_view_weight * loss_inv_view + self.args.alpha * loss_intra_view #loss_intra #+ self.args.beta * (recons_err + recons_nei)

            # compute hessian (finite difference approximation)
            model_weights_ = tuple(self.model_copy.parameters())
            d_model = torch.autograd.grad(loss, model_weights_)
            hessian = self.compute_hessian(d_model, features, adj_list, self.idx_p_list, epoch, self.knn_edge)
            # print(hessian)

            # update final gradient = - alpha * hessian
            with torch.no_grad():
                for mw, h in zip(self.meta_model.parameters(), hessian):
                    mw.grad = - 0.1 * h

            # optimiser_meta.step()

            # loss.backward()
            # optimiser.step()
        # torch.save(model.state_dict(), 'saved_model/best_{}_{}.pkl'.format(self.args.dataset,self.args.custom_key))
        # if self.args.use_pretrain:
        #     model.load_state_dict(torch.load('saved_model/best_{}_{}.pkl'.format(self.args.dataset,self.args.custom_key)))
        # print('loss', loss)
        print("Evaluating...")
        self.model.eval()
        self.meta_model.eval()
        h_p_list = self.model.embed(features, adj_list)
        hf = self.meta_model.embed(h_p_list, adj_list, h_a_list, self.knn_edge)
        # h_concat.append(embs_het)
        # h_concat.append(emb_hom)
        # hf = torch.cat(h_p_list, 1)

        # h_fusion = self.combine_att(h_p_list)
        macro_f1s, micro_f1s = evaluate(hf, self.idx_train, self.idx_val, self.idx_test, self.labels,task=self.args.custom_key,epoch = self.args.test_epo,lr = self.args.test_lr,iterater=self.args.iterater) #,seed=seed
        return macro_f1s, micro_f1s

    def unrolled_backward(self, features, adj_list, idx_p_list, optimiser, optimiser_meta, epoch):
        """
        Compute un-rolled loss and backward its gradients
        """

        #  compute unrolled multi-task network theta_1^+ (virtual step)


        x_list, h_p_list = self.model(features, adj_list, self.idx_p_list, epoch) #, recons
        # h_p_list, _ = self.meta_model(h_p_list)
        loss_intra = 0
        loss_inv = 0
        # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
        for i in range(self.args.view_num):
            intra_c = (h_p_list[i]).T @ (h_p_list[i])
            intra_c = F.normalize(intra_c, p=2, dim=1)
            on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
            off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
            loss_intra += (on_diag_intra + self.args.lambdintra[i] * off_diag_intra)
            if i == 1 and self.args.view_num == 2:
                break
            inter_c = (h_p_list[(i + 1) % self.args.view_num]).T @ (h_p_list[i])
            inter_c = F.normalize(inter_c, p=2, dim=1)
            loss_inv += -torch.diagonal(inter_c).sum()
        loss = loss_inv + self.args.alpha * loss_intra #+ self.args.beta * (recons_err + recons_nei)


        alpha = 0.1

        gradients = torch.autograd.grad(loss, self.model.parameters())
        # gradients = copy.deepcopy(
        #     [v.grad.data if v.grad is not None else None for v in self.model.parameters()])

        # gradients = copy.deepcopy(
        #     [v.grad.data if v.grad is not None else None for v in self.model.parameters()])

        # do virtual step: theta_1^+ = theta_1 - alpha * (primary loss + auxiliary loss)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.parameters(), self.model_copy.parameters(),
                                             gradients):
                if 'momentum' in optimiser.param_groups[0].keys():  # used in SGD with momentum
                    if optimiser.param_groups[0]['momentum'] == 0:
                        m = 0
                    else:
                        m = optimiser.state[weight].get('momentum_buffer', 0.) * optimiser.param_groups[0]['momentum']
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + optimiser.param_groups[0]['weight_decay'] * weight))

        # meta-training step: updating theta_2
        x_list, h_p_list = self.model_copy(features, adj_list, self.idx_p_list, epoch) #, recons
        h_p_list, _ = self.meta_model(h_p_list)

        # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
        loss_inv, loss_intra = intra_inter_loss(h_p_list, self.args)
        loss = loss_inv + self.args.alpha * loss_intra #+ self.args.beta * (recons_err + recons_nei)


        # compute hessian (finite difference approximation)
        model_weights_ = tuple(self.model_copy.parameters())
        d_model = torch.autograd.grad(loss, model_weights_)
        hessian = self.compute_hessian(d_model, features, adj_list, self.idx_p_list, epoch)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip(self.label_generator.parameters(), hessian):
                mw.grad = - alpha * h



    def compute_hessian(self, d_model, features, adj_list, idx_p_list, epoch , knn_edge):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm
        s = torch.zeros((features.shape[0], features.shape[0])).to(self.args.device)

        # theta_1^l = theta_1 + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        x_list, h_p_list, h_a_list = self.model(features, adj_list, idx_p_list, epoch) #, recons
        fusion_emb, node_weight, g_score, edge_weight = self.meta_model(h_p_list, adj_list, h_a_list, knn_edge)

        # s[knn_edge[0], knn_edge[1]] = edge_weight
        # h_new = torch.mm(edge_weight, h_a_list[0])
        # fusion_emb = torch.cat([fusion_emb, h_new], dim=1)

        # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
        loss_inv_view, loss_intra_view = View_intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
        loss_inv, loss_intra = intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
        loss = loss_inv + self.args.inv_view_weight * loss_inv_view + self.args.alpha * loss_intra_view  # loss_intra #+ self.args.beta * (recons_err + recons_nei)


        d_weight_p = torch.autograd.grad(loss, self.meta_model.parameters(), create_graph=True)

        # theta_1^r = theta_1 - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        x_list, h_p_list, h_a_list = self.model(features, adj_list, idx_p_list, epoch) #, recons
        fusion_emb, node_weight, g_score, edge_weight = self.meta_model(h_p_list, adj_list, h_a_list, knn_edge)

        # s[knn_edge[0], knn_edge[1]] = edge_weight
        # h_new = torch.mm(edge_weight, h_a_list[0])
        # fusion_emb = torch.cat([fusion_emb, h_new], dim=1)

        # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
        loss_inv_view, loss_intra_view = View_intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
        loss_inv, loss_intra = intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
        loss = loss_inv + self.args.inv_view_weight * loss_inv_view + self.args.alpha * loss_intra_view  # loss_intra #+ self.args.beta * (recons_err + recons_nei)


        d_weight_n = torch.autograd.grad(loss, self.meta_model.parameters(), create_graph=True)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian

    # def per_optimizer_step(self,
    #                     optimizer_a=None,
    #                     optimizer_b=None,
    #                     loss=None):
    #
    #     # update params
    #     if loss is not None:
    #         optimizer_a.zero_grad()
    #         if optimizer_b is not None:
    #             optimizer_b.zero_grad()
    #         loss.backward()
    #
    #     optimizer_a.step()
    #     optimizer_a.zero_grad()
    #     if optimizer_b is not None:
    #         optimizer_b.step()
    #         optimizer_b.zero_grad()

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

def cos_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

import multiprocessing as mp
class knn():
    def __init__(self, feats, k, index_path='', verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return (th_nbrs, th_dists)

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        # with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
        #            self.verbose):
        self.th = th
        self.th_knns = []
        tot = len(self.knns)
        if nproc > 1:
            pool = mp.Pool(nproc)
            th_knns = list(
                tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
            pool.close()
        else:
            th_knns = [self.filter_by_th(i) for i in range(tot)]
        return th_knns


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def View_intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, args):
    loss_intra = 0
    loss_inv = 0
    for i in range(args.view_num):
        intra_c = (h_p_list[i]).T @ (h_p_list[i])
        intra_c = F.normalize(intra_c, p=2, dim=1)
        on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
        off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
        loss_intra += (on_diag_intra + args.lambdintra[i] * off_diag_intra)
        if i == 1 and args.view_num == 2:
            break
        inter_c = (h_p_list[(i + 1) % args.view_num]).T @ (h_p_list[i])
        # inter_c = F.normalize(inter_c, p=2, dim=1)
        # expanded_weights = np.tile(node_weight, (1, embeddings.shape[1]))
        # inter_c = (fusion_emb * g_score.squeeze(0)[i] * node_weight.T.squeeze()[i].unsqueeze(1) * args.view_num * args.view_num).T @ (h_p_list[i])
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_inv += -torch.diagonal(inter_c).sum()
        # loss_intra = 0
    # intra_c = fusion_emb.T @ fusion_emb
    # intra_c = F.normalize(intra_c, p=2, dim=1)
    # on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
    # off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
    # loss_intra = (on_diag_intra + args.lambdintra[0] * off_diag_intra) #on_diag_intra +
    # loss_intra =

    return loss_inv, loss_intra

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear1 = Linearlayer(2,args.cfg[-1]*2, args.cfg[-1], args.ft_size)
        self.linear2 = nn.Linear(args.ft_size, args.ft_size)

    def forward(self, emb):
        recons = self.linear1(emb)
        recons = self.linear2(F.relu(recons))
        return recons

class trainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # cfg = args.cfg
        # view_num = args.view_num
        # self.w_list = nn.ModuleList([nn.Linear(cfg[-1], cfg[-1], bias=True) for _ in range(view_num)])
        # self.y_list = nn.ModuleList([nn.Linear(cfg[-1], 1) for _ in range(view_num)])
        # self.W = nn.Parameter(torch.zeros(size=(view_num * cfg[-1], cfg[-1])))
        # self.att_act1 = nn.Tanh()
        # self.att_act2 = nn.Softmax(dim=-1)
        self.encoder = nn.ModuleList()
        self.deg_enc_list = []
        # self.decoder = nn.ModuleList()
        for i in range(self.args.view_num):
            self.encoder.append(make_mlplayers(self.args.ft_size, self.args.cfg))
            # self.decoder.append(Decoder(args))
            self.deg_enc_list.append(nn.Embedding(int(max(self.args.degree_list[i]) + 1), self.args.cfg[-1], padding_idx=0)) #self.args.cfg[-1]
        # for i in range(view_num):

            # self.decoder.append(Decoder(args))

    def decode(self, embedding_list):
        recons = []
        for i in range(self.args.view_num):
            tmp = self.decoder[i](embedding_list[i])
            recons.append(tmp)

        return recons

    # def combine_att(self, h_list):
    #     h_combine_list = []
    #     for i, h in enumerate(h_list):
    #         h = self.w_list[i](h)
    #         h = self.y_list[i](h)
    #         h_combine_list.append(h)
    #     score = torch.cat(h_combine_list, -1)
    #     score = self.att_act1(score)
    #     score = self.att_act2(score)
    #     score = torch.unsqueeze(score, -1)
    #     h = torch.stack(h_list, dim=1)
    #     h = score * h
    #     h = torch.sum(h, dim=1)
    #     return h

    def forward(self, x, adj_list=None, idx_p_list=None, epoch=0):
        x = F.dropout(x, self.args.dropout, training=self.training)
        x_list = []
        for i in range(self.args.view_num):
            x_list.append(x)
        h_p_list = []
        h_a_list = []
        for i in range(self.args.view_num):
            h_a = self.encoder[i](x)
            deg_encoding = self.deg_enc_list[i](torch.LongTensor(self.args.degree_list[i])).to(self.args.device)
            # feature_encoding = self.Linear(h_list[i])
            # deg_enc_list.append(deg_encoding)
            h_a = torch.cat([h_a, deg_encoding], dim=1)
            h_a_list.append(h_a)

            if self.args.sparse:
                h_p = torch.spmm(adj_list[i], h_a)
            else:
                h_p = torch.mm(adj_list[i], h_a)
            h_p_list.append(h_p) #不做卷积或者做

        # recons = self.decode(h_p_list)

        return  x_list, h_p_list, h_a_list#, recons

    def embed(self, x, adj_list=None):
        h_p_list = []
        # embedding = []
        for i in range(self.args.view_num):
            h_a = self.encoder[i](x)
            deg_encoding = self.deg_enc_list[i](torch.LongTensor(self.args.degree_list[i])).to(self.args.device)
            # feature_encoding = self.Linear(h_list[i])
            # deg_enc_list.append(deg_encoding)
            h_a = torch.cat([h_a, deg_encoding], dim=1)
            if self.args.sparse:
                h_p = torch.spmm(adj_list[i], h_a)
            else:
                h_p = torch.mm(adj_list[i], h_a)
            h_p_list.append(h_p.detach()) #不做卷积或者做


        return h_p_list

# def intra_inter_loss(h_p_list, fusion_emb, args):
#     loss_intra = 0
#     loss_inv = 0
#     for i in range(args.view_num):
#         intra_c = (h_p_list[i]).T @ (h_p_list[i])
#         intra_c = F.normalize(intra_c, p=2, dim=1)
#         on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
#         off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
#         loss_intra += (on_diag_intra + args.lambdintra[i] * off_diag_intra)
#         if i == 1 and args.view_num == 2:
#             break
#         inter_c = (h_p_list[(i + 1) % args.view_num]).T @ (h_p_list[i])
#         inter_c = F.normalize(inter_c, p=2, dim=1)
#         loss_inv += -torch.diagonal(inter_c).sum()
#
#     return loss_inv, loss_intra

def intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, args):
    # loss_intra = 0
    loss_inv = 0
    for i in range(args.view_num):
        # intra_c = (h_p_list[i]).T @ (h_p_list[i])
        # intra_c = F.normalize(intra_c, p=2, dim=1)
        # on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
        # off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
        # loss_intra += (on_diag_intra + args.lambdintra[i] * off_diag_intra)
        # if i == 1 and args.view_num == 2:
        #     break
        # inter_c = fusion_emb.T @ (h_p_list[i])
        # expanded_weights = np.tile(node_weight, (1, embeddings.shape[1]))
        inter_c = (fusion_emb * g_score.squeeze(0)[i] * node_weight.T.squeeze()[i].unsqueeze(1) * args.view_num * args.view_num).T @ (h_p_list[i])
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_inv += -torch.diagonal(inter_c).sum()
    intra_c = fusion_emb.T @ fusion_emb
    intra_c = F.normalize(intra_c, p=2, dim=1)
    on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
    off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
    loss_intra = (on_diag_intra + args.lambdintra[0] * off_diag_intra) #on_diag_intra +
    # loss_intra =

    return loss_inv, loss_intra


# def View_intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, args):
#     loss_intra = 0
#     loss_inv = 0
#     for i in range(args.view_num):
#         intra_c = (h_p_list[i]).T @ (h_p_list[i])
#         intra_c = F.normalize(intra_c, p=2, dim=1)
#         on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
#         off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
#         loss_intra += (on_diag_intra + args.lambdintra[i] * off_diag_intra)
#         if i == 1 and args.view_num == 2:
#             break
#         inter_c = fusion_emb.T @ (h_p_list[i])
#         # expanded_weights = np.tile(node_weight, (1, embeddings.shape[1]))
#         # inter_c = (fusion_emb * g_score.squeeze(0)[i] * node_weight.T.squeeze()[i].unsqueeze(1) * args.view_num * args.view_num).T @ (h_p_list[i])
#         inter_c = F.normalize(inter_c, p=2, dim=1)
#         loss_inv += -torch.diagonal(inter_c).sum()
#     # intra_c = fusion_emb.T @ fusion_emb
#     # intra_c = F.normalize(intra_c, p=2, dim=1)
#     # on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
#     # off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
#     # loss_intra = (on_diag_intra + args.lambdintra[0] * off_diag_intra) #on_diag_intra +
#     # loss_intra =
#
#     return loss_inv, loss_intra



def loss_matching_recons(x_hat, x, idx_p_list, args, epoch):
    l = torch.nn.MSELoss(reduction='sum')

    recons_err = 0
    # Feature reconstruction loss
    for i in range(args.view_num):
        recons_err += l(x_hat[i], x[i])
    recons_err /= x[0].shape[0]

    # Topology reconstruction loss
    interval = int(args.neighbor_num / args.sample_neighbor)
    neighbor_embedding = []
    for i in range(args.view_num):
        neighbor_embedding_0 = []
        for j in range(0, args.sample_neighbor + 1):
            neighbor_embedding_0.append(x[i][idx_p_list[i][(epoch + interval * j) % args.neighbor_num]])
        neighbor_embedding.append(sum(neighbor_embedding_0) / args.sample_neighbor)
    recons_nei = 0
    for i in range(args.view_num):
        recons_nei += l(x_hat[i], neighbor_embedding[i])
    recons_nei /= x[0].shape[0]

    return recons_err, recons_nei


def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
            # result = nn.Sequential(*layers)
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)#, result

class Linearlayer(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(Linearlayer, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)