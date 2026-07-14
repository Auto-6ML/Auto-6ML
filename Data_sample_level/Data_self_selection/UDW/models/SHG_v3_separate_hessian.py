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
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from utils import process

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
        self.w_list = nn.ModuleList([nn.Linear(cfg[-1] + args.deg_dim, cfg[-1], bias=True) for _ in range(view_num)])
        self.y_list = nn.ModuleList([nn.Linear(cfg[-1], 1) for _ in range(view_num)])
        # self.map = nn.ModuleList([nn.Linear(cfg[-1] * 2, cfg[-1] * 2) for _ in range(view_num)])
        # self.W = nn.Parameter(torch.zeros(size=(view_num * cfg[-1], cfg[-1])))
        self.att_act1 = nn.Tanh()
        self.threshold = 0.2
        self.att_act2 = nn.Softmax(dim=-1)
        self.mlp_edge_model = Sequential(
            Linear(cfg[-1] * 2 + args.deg_dim*2 , 64),
            ReLU(),
            Linear(64, 1))
        self.disc = Discriminator(self.args.cfg[-1] + self.args.deg_dim)
        self.b_xent = nn.BCEWithLogitsLoss()
        # for i in range(view_num):
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
            # S_new = torch.where(S_new < self.threshold, torch.zeros_like(S_new), S_new)
            # S_new = torch.mul(S_new, adj_list[i])
            # h_new = torch.mm(S_new, h_a_list[i])
            #
            # # h = h + h_new
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
        bias = 0#0.0001
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        edge_score = torch.log(eps) - torch.log(1 - eps)
        edge_score = edge_score.to(self.args.device)
        edge_score = (edge_score + edge_logits)
        # edge_score = edge_logits
        edge_weight = torch.sigmoid(edge_score).squeeze()
        s[src, dst] = edge_weight

        # for i in range(self.args.view_num):
            # c = AvgReadout(h_p_list[i], msk=None)
        c = torch.mean(h, 1)
        c = torch.sigmoid(c)  # 获得公式中的S
        random = np.random.permutation(self.args.nb_nodes)  # np.random.randint(h_p_list[i].shape[0], h_p_list[i].shape[0])
        h_2 = h[random]

        # h_2 = self.gcn(seq2, adj, sparse)  # Hi~

        ret = self.disc(c, h.unsqueeze(1), h_2.unsqueeze(1), None, None)
        # logits = torch.cat((sc_1, sc_2), 1)

        lbl_1 = torch.ones(1, self.args.nb_nodes)
        lbl_2 = torch.zeros(1, self.args.nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)
        loss = self.b_xent(ret, lbl)
        # # s = (s + s.T)/2
        # #
        # h_new = torch.mm(s, feature_embedding)
        # #
        # # h = h +  h_new
        # h = torch.cat([h, 0.1 * h_new], dim=1)

        # value, index = torch.sort(edge_weight, descending=True)
        # value_new = value[index[int(index.shape[0]*0.6)]]
        # positive_index = index[value>value_new]
        # positive_edge = knn[0][index[value>value_new]], knn[1][index[value>value_new]]
        # negative_edge = knn[0][index[value < value_new]], knn[1][index[value < value_new]]




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



        return h, score, g_score, s, edge_weight, loss
        #############################################

    def embed(self, h_list, adj_list, h_a_list,knn):
        h_combine_list = []
        for i, h in enumerate(h_list):
            # deg_encoding = self.deg_enc_list[i](torch.LongTensor(self.args.degree_list[i])).to(self.args.device)
            # deg_enc_list.append(deg_encoding)
            # h = torch.cat([h, deg_encoding], dim=1)
            # h_a_map = self.map[i](h_a_list[i])
            # S_new = cosine_dist(h_a_map, h_a_map)
            # S_new = torch.where(S_new < self.threshold, torch.zeros_like(S_new), S_new)
            # S_new = torch.mul(S_new, adj_list[i])
            # h_new = torch.mm(S_new, h_a_list[i])

            # h = h + h_new
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

        feature_embedding = sum(h_a_list) / self.args.view_num
        s = torch.zeros((h.shape[0], h.shape[0])).to(self.args.device)
        src, dst = knn
        emb_src = feature_embedding[src]
        emb_dst = feature_embedding[dst]
        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)
        bias = 0 #0.0001
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        edge_score = torch.log(eps) - torch.log(1 - eps)
        edge_score = edge_score.to(self.args.device)
        edge_score = (edge_score + edge_logits)
        # edge_score = edge_logits
        edge_weight = torch.sigmoid(edge_score).squeeze()
        s[src, dst] = edge_weight
        # s = (s + s.T) / 2
        # #
        # h_new = torch.mm(s, feature_embedding)
        #
        # h = h +  h_new
        # h = torch.cat([h, 0.1 * h_new], dim=1)

        return h.detach()


def cosine_dist(x, y):
    bs1 = x.size()[0]
    bs2 = y.size()[0]
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2) + 0.00000001, 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2) + 0.00000001, 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return cosine

class SHG_v3_hessian(embedder):
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
        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        print("Started training...")

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        optimiser_meta = torch.optim.Adam(self.meta_model.parameters(), lr=self.args.meta_lr)
        self.adj_new = torch.zeros((self.args.nb_nodes, self.args.nb_nodes)).to(self.args.device)
        # b_xent = nn.BCEWithLogitsLoss()


        # self.adj_new =

        # self.model.train()

        for epoch in tqdm(range(self.args.nb_epochs+1)):
    ###################################train model########################################
            self.model.train()
            self.meta_model.eval()
            optimiser.zero_grad()

            for i in range(1):
                x_list, h_p_list, h_a_list = self.model(features, adj_list, self.idx_p_list, self.adj_new.detach(), epoch) #, recons
                fusion_emb, node_weight, g_score, adj_new, edge_weight, loss = self.meta_model(h_p_list, adj_list, h_a_list,
                                                                                         self.knn_edge)
                # loss = ret

                # s[src, dst] = edge_weight
                # fusion_emb, node_weight, g_score, edge_weight = self.meta_model(h_p_list, adj_list, h_a_list, self.knn_edge)
                # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
                # loss_inv, loss_intra = intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)

                # loss_spec = spectral_loss(h_p_list, self.args)

                loss_inv_view, loss_intra_view, std_loss = View_intra_inter_loss(h_p_list, node_weight, g_score, self.args)
                # loss_inv_view, loss_intra_view, std_loss = View_intra_inter_loss(h_p_list,
                #                                                                  self.args)
                # loss_disc = DGI_loss(h_p_list,  self.args)
                # vicreg_loss = vic_loss(h_p_list,  self.args)
                # loss_inv, loss_intra = intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
                # sample__loss = sample_decorrelation_loss(fusion_emb, edge_weight, self.knn_edge, self.args)
                # loss_edge = loss_contrastive(fusion_emb, edge_weight, self.knn_edge, self.args)
                loss = loss_inv_view + self.args.alpha * loss_intra_view + self.args.std_w * std_loss#+ self.args.con * sample__loss #(loss_inv_view + self.args.alpha * loss_intra_view)# + self.args.con * loss_edge loss_inv + self.args.fusionintra * loss_intra + s
                # loss =  loss_spec #+ self.args.alpha * loss_intra_view#+ loss_inv + self.args.fusionintra * loss_intra + self.args.con * sample__loss


                loss.backward()
                optimiser.step()

##############################################train meta##############################################
            self.model.eval()
            self.meta_model.train()
            optimiser_meta.zero_grad()

            # x_list, h_p_list, h_a_list = self.model(features, adj_list, self.idx_p_list, epoch) #, recons
            # fusion_emb, node_weight, g_score, edge_weight = self.meta_model(h_p_list, adj_list, h_a_list, self.knn_edge)
            # # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
            # # loss_inv, loss_intra = intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
            #
            # # loss_inv, loss_intra = View_intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
            #
            # loss_spec = spectral_loss(h_p_list, self.args)
            # loss = loss_spec# loss_inv + self.args.alpha * loss_intra #+ self.args.beta * (recons_err + recons_nei)
            #
            # # current parameter
            # # fast_weights = OrderedDict((name, param) for (name, param) in self.model.named_parameters())
            #
            # # create_graph flag for computing second-derivative
            # grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # # data = [p.data for p in list(self.model.parameters())]
            #
            # with torch.no_grad():
            #     for weight, weight_, grad in zip(self.model.parameters(), self.model_copy.parameters(),
            #                                      grads):
            #         if 'momentum' in optimiser.param_groups[0].keys():  # used in SGD with momentum
            #             if optimiser.param_groups[0]['momentum'] == 0:
            #                 m = 0
            #             else:
            #                 m = optimiser.state[weight].get('momentum_buffer', 0.) * optimiser.param_groups[0]['momentum']
            #         else:
            #             m = 0
            #         weight_.copy_(weight - 0.1 * (m + grad + optimiser.param_groups[0]['weight_decay'] * weight))



            # compute parameter' by applying sgd on multi-task loss
            # fast_weights = OrderedDict(
            #     (name, param - 0.1 * grad) for ((name, param), grad, data)  #args.LGA_lr
            #     in zip(fast_weights.items(), grads, data))  # \fi0 and \varfi0

            # compute primary loss with the updated parameter'

            x_list, h_p_list, h_a_list = self.model(features, adj_list, self.idx_p_list, self.adj_new.detach(), epoch) #, recons
            fusion_emb, node_weight, g_score, adj_new,  edge_weight, loss_dgi= self.meta_model(h_p_list, adj_list, h_a_list, self.knn_edge) #, edge_weight

            # self.adj_new[self.knn_edge[0], self.knn_edge[1]] = edge_weight


            # torch.cuda.empty_cache()
            # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
            # loss_inv, loss_intra = intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)

            loss_inv_view, loss_intra_view, std_loss = View_intra_inter_loss(h_p_list, node_weight, g_score, self.args)
            loss = loss_inv_view + self.args.alpha * loss_intra_view + self.args.std_w * std_loss

            # loss_disc = DGI_loss_fusion(fusion_emb, self.args)
            # loss_edge = loss_contrastive(fusion_emb, edge_weight, self.knn_edge, self.args)

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

            x_list, h_p_list, h_a_list = self.model_copy(features, adj_list, self.idx_p_list, self.adj_new.detach(),
                                                    epoch)  # , recons
            fusion_emb, node_weight, g_score, adj_new, edge_weight, loss_dgi = self.meta_model(h_p_list, adj_list, h_a_list,
                                                                                       self.knn_edge)

            self.adj_new = adj_new

            sample_loss = sample_decorrelation_loss(fusion_emb, edge_weight, self.knn_edge, self.args)

            loss = loss_dgi + self.args.con * sample_loss

            loss.backward()
            optimiser_meta.step()
            torch.cuda.empty_cache()

            # s[knn_edge[0], knn_edge[1]] = edge_weight
            # h_new = torch.mm(edge_weight, h_a_list[0])
            # fusion_emb = torch.cat([fusion_emb, h_new], dim=1)

            # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
            # loss_inv_view, loss_intra_view = View_intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
            # loss_inv, loss_intra = intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
            # loss = loss_inv + self.args.inv_view_weight * loss_inv_view + self.args.alpha * loss_intra_view  # loss_intra #+ self.args.beta * (recons_err + recons_nei)
            #
            # # compute hessian (finite difference approximation)
            # model_weights_ = tuple(self.model_copy.parameters())
            # d_model = torch.autograd.grad(loss, model_weights_)
            # hessian = self.compute_hessian(d_model, features, adj_list, self.idx_p_list, epoch, self.knn_edge)
            # # print(hessian)
            #
            # # update final gradient = - alpha * hessian
            # with torch.no_grad():
            #     for mw, h in zip(self.meta_model.parameters(), hessian):
            #         mw.grad = - 0.1 * h

                    # loss = loss_dgi  + self.args.con * sample_loss #+ self.args.con * loss_edge  #+ self.args.beta * (recons_err + recons_nei) + self.args.fusionintra * loss_intra

                    # loss = loss_inv + self.args.fusionintra * loss_intra



            # compute hessian (finite difference approximation)
            # model_weights_ = tuple(self.model_copy.parameters())
            # d_model = torch.autograd.grad(loss, model_weights_)
            # hessian = self.compute_hessian(d_model, features, adj_list, self.idx_p_list, epoch)
            # print(hessian)

            # update final gradient = - alpha * hessian
            # with torch.no_grad():
            #     for mw, h in zip(self.meta_model.parameters(), hessian):
            #         mw.grad = - 0.1 * h

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
        h_p_list = self.model.embed(features, adj_list, self.adj_new)
        hf = self.meta_model.embed(h_p_list, adj_list, h_a_list, self.knn_edge)
        # h_concat.append(embs_het)
        # h_concat.append(emb_hom)
        # hf = torch.cat(h_p_list, 1)

        # h_fusion = self.combine_att(h_p_list)
        macro_f1s, micro_f1s = evaluate(hf, self.idx_train, self.idx_val, self.idx_test, self.labels,task=self.args.custom_key,epoch = self.args.test_epo,lr = self.args.test_lr,iterater=self.args.iterater) #,seed=seed
        return macro_f1s, micro_f1s



    def compute_hessian(self, d_model, features, adj_list, idx_p_list, epoch):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # theta_1^l = theta_1 + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        x_list, h_p_list, h_a_list = self.model(features, adj_list, idx_p_list, epoch) #, recons
        fusion_emb, node_weight, g_score = self.meta_model(h_p_list, adj_list, h_a_list)
        # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
        loss_inv, loss_intra = intra_inter_loss(h_p_list, fusion_emb, node_weight, g_score, self.args)
        loss = loss_inv + self.args.alpha * loss_intra #+ self.args.beta * (recons_err + recons_nei)


        d_weight_p = torch.autograd.grad(loss, self.meta_model.parameters(), create_graph=True)

        # theta_1^r = theta_1 - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        x_list, h_p_list, h_a_list = self.model(features, adj_list, idx_p_list, epoch) #, recons
        fusion_emb, node_weight, g_score = self.meta_model(h_p_list, adj_list, h_a_list)
        # recons_err, recons_nei = loss_matching_recons(recons, x_list, self.idx_p_list, self.args, epoch)
        loss_inv, loss_intra = intra_inter_loss(h_p_list,fusion_emb, node_weight, g_score, self.args)
        loss = loss_inv + self.args.alpha * loss_intra #+ self.args.beta * (recons_err + recons_nei)


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


def View_intra_inter_loss(h_p_list, node_weight, g_score, args):
    loss_intra = 0
    loss_inv = 0
    repr_loss = 0
    for i in range(args.view_num):
        intra_c = (h_p_list[i]).T @ (h_p_list[i])
        intra_c = F.normalize(intra_c, p=2, dim=1)
        on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
        off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
        loss_intra += (on_diag_intra + args.lambdintra[i] * off_diag_intra)
        if i == 1 and args.view_num == 2:
            break
        # inter_c = (h_p_list[(i + 1) % args.view_num]).T @ (h_p_list[i])
        # inter_c = F.normalize(inter_c, p=2, dim=1)
        # expanded_weights = np.tile(node_weight, (1, embeddings.shape[1]))
        inter_c = (h_p_list[(i + 1) % args.view_num] * g_score.squeeze(0)[(i + 1) % args.view_num] * node_weight.T.squeeze()[(i + 1) % args.view_num].unsqueeze(1) * args.view_num * args.view_num).T @ (h_p_list[i]* g_score.squeeze(0)[i] * node_weight.T.squeeze()[i].unsqueeze(1) * args.view_num * args.view_num)
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_inv += -torch.diagonal(inter_c).sum()
        repr_loss += F.mse_loss(h_p_list[i], h_p_list[(i + 1) % args.view_num])

        std_x = torch.sqrt(h_p_list[i].var(dim=0) + 0.0001)
        std_y = torch.sqrt(h_p_list[(i + 1) % args.view_num].var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # cov_x = (h_p_list[i].T @ h_p_list[i]) / (args.nb_nodes - 1)
        # cov_y = (h_p_list[(i + 1) % args.view_num].T @ h_p_list[(i + 1) % args.view_num]) / (args.nb_nodes - 1)
        # cov_loss = off_diagonal(cov_x).pow_(2).sum().div(args.cfg[-1]) + off_diagonal(cov_y).pow_(2).sum().div(
        #     args.cfg[-1])


        # loss_intra = 0
    # intra_c = fusion_emb.T @ fusion_emb
    # intra_c = F.normalize(intra_c, p=2, dim=1)
    # on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
    # off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
    # loss_intra = (on_diag_intra + args.lambdintra[0] * off_diag_intra) #on_diag_intra +
    # loss_intra =

    return loss_inv, loss_intra, std_loss

def DGI_loss(h_p_list, args):
    ret = 0
    for i in range(args.view_num):
        # c = AvgReadout(h_p_list[i], msk=None)
        c = torch.mean(h_p_list[i], 1)
        c = torch.sigmoid(c)  # 获得公式中的S
        random = np.random.permutation(args.nb_nodes)#np.random.randint(h_p_list[i].shape[0], h_p_list[i].shape[0])
        h_2 = h_p_list[i][random]

        # h_2 = self.gcn(seq2, adj, sparse)  # Hi~

        ret += Discriminator(c, h_p_list[i].unsqueeze(1), h_2.unsqueeze(1), None, None)

    return ret


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        #  c = self.read(h_1, msk)
        if msk is None:
            return torch.mean(seq, 1)                            #R函数
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

def vic_loss(h_p_list, args):
    for i in range(args.view_num):
        if i == 1 and args.view_num == 2:
            break
        repr_loss = F.mse_loss(h_p_list[i], h_p_list[(i + 1) % args.view_num])

        x = h_p_list[i] - h_p_list[i].mean(dim=0)
        y = h_p_list[(i + 1) % args.view_num] - h_p_list[(i + 1) % args.view_num].mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (args.nb_nodes - 1)
        cov_y = (y.T @ y) / (args.nb_nodes - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(args.cfg[-1]) + off_diagonal(cov_y).pow_(2).sum().div(args.cfg[-1])

        loss = (100 * repr_loss + 100 * std_loss + 1 * cov_loss)
    return loss

def spectral_loss(h_p_list, args, mu=1.0):
    loss_part1 = 0
    loss_part2 = 0
    for i in range(args.view_num):
        mask1 = (torch.norm(h_p_list[i], p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        mask2 = (torch.norm(h_p_list[(i + 1) % args.view_num], p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        # z1 = mask1 * h_p_list[i] + (1-mask1) * F.normalize(h_p_list[i], dim=1) * np.sqrt(mu)
        # z2 = mask2 * h_p_list[(i + 1) % args.view_num] + (1-mask2) * F.normalize(h_p_list[(i + 1) % args.view_num], dim=1) * np.sqrt(mu)
        z1 = mask1 * h_p_list[i] + (1-mask1) * h_p_list[i] * np.sqrt(mu)
        z2 = mask2 * h_p_list[(i + 1) % args.view_num] + (1-mask2) * h_p_list[(i + 1) % args.view_num] * np.sqrt(mu)
        # z1 = h_p_list[i]
        # z2 = h_p_list[(i + 1) % args.view_num]
        loss_part1 += -2 * torch.mean(z1 * z2) * z1.shape[1]
        square_term = torch.matmul(z1, z2.T) ** 2
        loss_part2 += torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                     z1.shape[0] / (z1.shape[0] - 1)
    return (loss_part1 + loss_part2) / mu#, {"part1": loss_part1 / mu, "part2": loss_part2 / mu}

def sample_decorrelation_loss(fusion_emb, edge_weight, knn, args):
    value, index = torch.sort(edge_weight, descending=True)
    # value_positive = value[index[int(index.shape[0] * 0.005)]]
    # value_negative = value[index[int(index.shape[0] * 0.999)]]
    value_positive = value[int(index.shape[0] * args.positive_rate)]
    value_negative = value[int(index.shape[0] * (1-args.positive_rate))]
    # positive_index = index[value > value_new]
    # positive_edge = knn[0][index[value > value_new]], knn[1][index[value > value_new]]
    # negative_edge = knn[0][index[value < value_new]], knn[1][index[value < value_new]]
    # # for adj in adj_list:
    # #     adj = adj_list[i]
    # out_node = adj.to_sparse()._indices()[1]
    # random = np.random.randint(out_node.shape[0], size=int((out_node.shape[0] / args.sample_num)))
    # sample_edge = adj.to_sparse()._indices().T[random]
    # dis = F.cosine_similarity(U[sample_edge.T[0]],U[sample_edge.T[1]])
    # a, maxidx = torch.sort(dis, descending=True)
    # idx1 = maxidx[:int(sample_edge.shape[0]*0.2)]
    # b, minidx = torch.sort(dis, descending=False)
    # idx2 = minidx[:int(sample_edge.shape[0]*0.1)]
    positive_index = index[value > value_positive]
    negative_index = index[value < value_negative]
    random_positive = np.random.randint(positive_index.shape[0], size=int(positive_index.shape[0] * 0.02))
    random_negative = np.random.randint(negative_index.shape[0], size=int(negative_index.shape[0] * 0.02))
    # sample_edge_positive = torch.random.choice(positive_index, positive_index.shape[0] * 0.1, replace=False)

    # sample_edge_negative = torch.random.choice(negative_index, positive_index.shape[0] * 0.1, replace=False)
    private_sample_0 = fusion_emb[knn[0][positive_index[random_positive]]]
    private_sample_1 = fusion_emb[knn[1][positive_index[random_positive]]]
    private_sample_2 = fusion_emb[knn[0][negative_index[random_negative]]]
    private_sample_3 = fusion_emb[knn[1][negative_index[random_negative]]]

    intra_c_positive = (private_sample_0) @ (private_sample_1).T
    # intra_c_positive = F.normalize(intra_c_positive, p=2, dim=1)
    inter_c_positive = F.normalize(intra_c_positive, p=2, dim=1)
    loss_inv_positive = -torch.diagonal(inter_c_positive).sum()

    inter_c_negative = (private_sample_2) @ (private_sample_3).T
    inter_c_negative = F.normalize(inter_c_negative, p=2, dim=1)
    # inter_c_negative = F.normalize(intra_c_negative, p=2, dim=1)
    loss_inv_negative = torch.diagonal(inter_c_negative).sum()

    return (loss_inv_positive + 1 * loss_inv_negative) #/ mu#, {"part1": loss_part1 / mu, "part2": loss_part2 / mu}



def loss_contrastive(fusion_emb, edge_weight, knn, args):
    i = 0
    loss = 0
    value, index = torch.sort(edge_weight, descending=True)
    value_new = value[index[int(index.shape[0] * 0.7)]]
    # positive_index = index[value > value_new]
    # positive_edge = knn[0][index[value > value_new]], knn[1][index[value > value_new]]
    # negative_edge = knn[0][index[value < value_new]], knn[1][index[value < value_new]]
    # # for adj in adj_list:
    # #     adj = adj_list[i]
    # out_node = adj.to_sparse()._indices()[1]
    # random = np.random.randint(out_node.shape[0], size=int((out_node.shape[0] / args.sample_num)))
    # sample_edge = adj.to_sparse()._indices().T[random]
    # dis = F.cosine_similarity(U[sample_edge.T[0]],U[sample_edge.T[1]])
    # a, maxidx = torch.sort(dis, descending=True)
    # idx1 = maxidx[:int(sample_edge.shape[0]*0.2)]
    # b, minidx = torch.sort(dis, descending=False)
    # idx2 = minidx[:int(sample_edge.shape[0]*0.1)]
    private_sample_0 = fusion_emb[knn[0][index[value > value_new]]]
    private_sample_1 = fusion_emb[knn[1][index[value > value_new]]]
    private_sample_2 = fusion_emb[knn[0][index[value < value_new]]]
    private_sample_3 = fusion_emb[knn[1][index[value < value_new]]]
    i += 1
    loss += semi_loss(private_sample_0, private_sample_1, private_sample_2, private_sample_3, args)

    return loss

def semi_loss(z1, z2, z3, z4, args):
    f = lambda x: torch.exp(x / args.tau)
    positive = f(F.cosine_similarity(z1, z2))
    negative = f(F.cosine_similarity(z3, z4))
    return -torch.log(
        positive.sum()
        / (positive.sum() + negative.sum() ))

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

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        c_x = torch.unsqueeze(c, 1)
        c_x = torch.unsqueeze(c_x, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)    #squeeze删去第三维, HWS
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)    #H~WS

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1.T, sc_2.T), 1)          #按列拼接

        return logits




def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear1 = Linearlayer(2,args.cfg[-1], args.cfg[-1], args.ft_size)
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
            self.deg_enc_list.append(nn.Embedding(int(max(self.args.degree_list[i]) + 1), self.args.deg_dim, padding_idx=0))
        # self.g = nn.Sequential(nn.Linear(self.args.cfg[-1] * 2, self.args.cfg[-1]* 2, bias=False),
        #                        nn.ReLU(inplace=True)).to(self.args.device)
        # self.disc = Discriminator(self.args.cfg[-1] + self.args.deg_dim)
        # self.b_xent = nn.BCEWithLogitsLoss()
        # for i in range(view_num):

            # self.decoder.append(Decoder(args))

    def decode(self, embedding_list):
        recons = []
        for i in range(self.args.view_num):
            tmp = self.decoder[i](embedding_list[i])
            recons.append(tmp)

        return recons

    def forward(self, x, adj_list,  idx_p_list, adj_new, epoch=0):
        x = F.dropout(x, self.args.dropout, training=self.training)
        x_list = []
        adj_new = process.normalize_graph(adj_new + torch.eye(self.args.nb_nodes, self.args.nb_nodes).to(self.args.device))
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
                h_p = torch.spmm((adj_list[i] + self.args.adj_weight * adj_new), h_a)
                # h_p = self.g(h_p)
            else:
                h_p = torch.mm((adj_list[i] + self.args.adj_weight * adj_new) , h_a)
                # h_p = self.g(h_p)
            h_p_list.append(h_p) #不做卷积或者做

        # loss = 0
        # for i in range(self.args.view_num):
        #     # c = AvgReadout(h_p_list[i], msk=None)
        #     c = torch.mean(h_p_list[i], 1)
        #     c = torch.sigmoid(c)  # 获得公式中的S
        #     random = np.random.permutation(self.args.nb_nodes)  # np.random.randint(h_p_list[i].shape[0], h_p_list[i].shape[0])
        #     h_2 = h_p_list[i][random]
        #
        #     # h_2 = self.gcn(seq2, adj, sparse)  # Hi~
        #
        #     ret = self.disc(c, h_p_list[i].unsqueeze(1), h_2.unsqueeze(1), None, None)
        #     # logits = torch.cat((sc_1, sc_2), 1)
        #
        #     lbl_1 = torch.ones(1, self.args.nb_nodes)
        #     lbl_2 = torch.zeros(1, self.args.nb_nodes)
        #     lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)
        #     loss += self.b_xent(ret, lbl)

        # recons = self.decode(h_p_list)

        return  x_list, h_p_list, h_a_list#, loss#, recons

    def embed(self, x,  adj_list, adj_new):
        h_p_list = []
        adj_new = process.normalize_graph(adj_new + torch.eye(self.args.nb_nodes, self.args.nb_nodes).to(self.args.device))
        # embedding = []
        for i in range(self.args.view_num):
            h_a = self.encoder[i](x)
            deg_encoding = self.deg_enc_list[i](torch.LongTensor(self.args.degree_list[i])).to(self.args.device)
            # feature_encoding = self.Linear(h_list[i])
            # deg_enc_list.append(deg_encoding)
            h_a = torch.cat([h_a, deg_encoding], dim=1)
            if self.args.sparse:
                h_p = torch.spmm((adj_list[i] + self.args.adj_weight * adj_new), h_a)
            else:
                h_p = torch.mm((adj_list[i] + self.args.adj_weight * adj_new), h_a)
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
        inter_c = fusion_emb.T @ (h_p_list[i])
        # expanded_weights = np.tile(node_weight, (1, embeddings.shape[1]))
        # inter_c = (fusion_emb * g_score.squeeze(0)[i] * node_weight.T.squeeze()[i].unsqueeze(1) * args.view_num * args.view_num).T @ (h_p_list[i])
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_inv += -torch.diagonal(inter_c).sum()
    intra_c = fusion_emb.T @ fusion_emb
    intra_c = F.normalize(intra_c, p=2, dim=1)
    on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
    off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
    loss_intra = (on_diag_intra + args.fusionintra * off_diag_intra) #on_diag_intra +
    # loss_intra = 0
    # loss_intra =

    return loss_inv, loss_intra



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