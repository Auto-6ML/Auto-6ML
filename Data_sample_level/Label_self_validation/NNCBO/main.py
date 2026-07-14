import warnings
import torch
import torch.nn.functional as F
import random
import argparse
import numpy as np
import yaml
from model.GCN import GCNNet
from model.teachers_weight_matrix import weight_matrix
from model.classifier import classifier
from utils.load_data import load_data
from utils.train_test import train_model
from utils.add_noisy import noisify_with_P
from utils.label_correction import sim, pre, select_clean, select_pseudo, select_noisy
from encoder.teacher1.get_embed1 import get_embed1
from encoder.teacher2.get_embed2 import get_embed2
from encoder.teacher3.get_embed3 import get_embed3

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def arg_parse(dataset_name, noise_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='FewGraph arguments.')
    parser.add_argument('--dataset_name', type=str, help='Cora/CiteSeer/Pubmed/cs/Computers/Photo/dblp'),
    parser.add_argument('--lr', type=int, default=0.01, help='learning rate')
    parser.add_argument('--epoch', type=int, help='max epoch')
    parser.add_argument('--no_progress', type=int, help='How many epochs no progress then break training')
    parser.add_argument('--noisy_rate', type=float, help='noisy rate')
    parser.add_argument('--random_seed', type=int, help='random seed')
    parser.add_argument('--pseudo_num', type=int,
                        help='How many pseudo tags are selected for each category in each round')
    parser.add_argument('--device', type=str, help='cuda or cpu')
    parser.add_argument('--iteration', type=int, help='iteration rounds')
    parser.add_argument('--noise_type', type=str, default='uniform', choices=['uniform', 'pair'],
                        help='type of noises')
    parser.add_argument('--noise_seed', type=int, default=0)
    parser.add_argument('--noise_rate', type=list, default=[0])
    parser.add_argument('--Matrix_lr', type=float, default=0.01)

    parser.set_defaults(
        dataset_name=dataset_name,
        epoch=1000,
        no_progress=100,
        noisy_rate=0,
        random_seed=0,
        pseudo_num=30,
        device=device,
        noise_type=noise_type,
        iteration=5,
    )
    return parser.parse_args()


def main(args, noise_r):
    args.noisy_rate = noise_r

    '''加载数据'''
    setup_seed(0)
    dataset, data, idx_train = load_data(args.dataset_name, device=args.device)
    data.to(args.device)

    # 添加噪声标签
    nclass = dataset.num_classes
    train_labels = data.y[idx_train].cpu().numpy()
    noise_y = noisify_with_P(train_labels, nclass, args.noisy_rate, 20, args.noise_type)
    noise_y = torch.tensor(noise_y)
    data.y[idx_train] = noise_y.to(args.device)
    '''加载数据'''

    setup_seed(args.random_seed)

    '''实例学生模型'''
    student = GCNNet(dataset).to(args.device)
    optimizer_student = torch.optim.Adam(student.parameters(),
                                         lr=0.001 if args.dataset_name in ['Photo', 'Computers'] else 0.01)

    '''创建多教师分类器'''
    classfier_teacher1 = classifier(128, dataset.num_classes).to(args.device)
    optimizer_teacher1 = torch.optim.Adam(classfier_teacher1.parameters(), lr=0.01)
    classfier_teacher2 = classifier(128, dataset.num_classes).to(args.device)
    optimizer_teacher2 = torch.optim.Adam(classfier_teacher2.parameters(), lr=0.01)
    classfier_teacher3 = classifier(512, dataset.num_classes).to(args.device)
    optimizer_teacher3 = torch.optim.Adam(classfier_teacher3.parameters(), lr=0.01)

    best_acc = 0
    for iters in range(args.iteration):
        '''实例教师权重矩阵'''
        Weight_matrix = weight_matrix(dataset.num_classes, 3).to(args.device)
        optimizer_Weight = torch.optim.Adam(Weight_matrix.parameters(), lr=args.Matrix_lr, weight_decay=0.01)

        '''训练多教师分类器'''
        classfier_teacher1 = train_model(classfier_teacher1, 'teacher1', embeds_1, data, optimizer_teacher1)
        classfier_teacher2 = train_model(classfier_teacher2, 'teacher2', embeds_2, data, optimizer_teacher2)
        classfier_teacher3 = train_model(classfier_teacher3, 'teacher3', embeds_3, data, optimizer_teacher3)

        pre_teacher1, pre_label_teacher1 = pre(classfier_teacher1, embeds_1, data)
        pre_teacher2, pre_label_teacher2 = pre(classfier_teacher2, embeds_2, data)
        pre_teacher3, pre_label_teacher3 = pre(classfier_teacher3, embeds_3, data)

        pre_add = F.softmax(pre_teacher1 + pre_teacher2 + pre_teacher3, dim=1)
        prelabel_idx, prelabel_idx_sim, prelabel_sim, select_clean_idx = select_clean(data, pre_label_teacher1,
                                                                                        pre_label_teacher2, pre_label_teacher3,
                                                                                        sim_add, pre_add)

        clean_val_mask = []
        for i in range(len(select_clean_idx)):
            for j in select_clean_idx[i]:
                clean_val_mask.append(j)
                data.y[j] = i

        train_mask_1 = []
        for i in range(data.size(0)):
            if data.val_mask[i] == 0 and data.test_mask[i] == 0:
                train_mask_1.append(i)

        '''Multi-Teacher Distillation Based on Bi-level Optimization'''
        student.eval()
        for i in range(30):
            Weight_matrix.train()
            optimizer_Weight.zero_grad()
            teacher_pre = Weight_matrix(pre_teacher1, pre_teacher2, pre_teacher3).to(args.device)
            student.train()

            pd_yt_x = torch.zeros([3, dataset.num_classes]).to(args.device)
            for j in range(5):
                student.train()
                loss_s_u = -(teacher_pre[train_mask_1] * F.log_softmax(student(data), dim=1)[train_mask_1]).mean()

                grad_1 = torch.autograd.grad(loss_s_u, student.parameters(), create_graph=True)
                W_grad = []
                inner_grad = []
                for grad in grad_1:
                    W_grad_grad = torch.autograd.grad(grad.sum(), Weight_matrix.parameters(), retain_graph=True)
                    if not W_grad:
                        W_grad.extend(W_grad_grad)
                    else:
                        for Wi in range(len(W_grad_grad)):
                            W_grad[Wi] += W_grad_grad[Wi]

                    inner_grad_grad = torch.autograd.grad(grad.sum(), student.parameters(), retain_graph=True)
                    if not inner_grad:
                        for ii in range(len(inner_grad_grad)):
                            inner_grad.append(1 - inner_grad_grad[ii])
                    else:
                        for ii in range(len(inner_grad_grad)):
                            inner_grad[ii] += (1 - inner_grad_grad[ii])

                if args.dataset_name in ['Photo', 'Computers']:
                    if j == 0:
                        pd_yt_x = (pd_yt_x * (
                                inner_grad[2] + (inner_grad[3] + inner_grad[0] + inner_grad[1].mean(1)).mean(
                            1))) + W_grad[0]
                    else:
                        pd_yt_x = (pd_yt_x * (
                                inner_grad[2] + (inner_grad[3] + inner_grad[0] + inner_grad[1].mean(1)).mean(
                            1))) * ((1 - j) / j) + W_grad[0] * (1 / j)
                else:
                    if j == 0:
                        pd_yt_x = (pd_yt_x * (
                                inner_grad[2] + (inner_grad[3] + inner_grad[0] + inner_grad[1].sum(1)).sum(
                            1))) + W_grad[0]
                    else:
                        pd_yt_x = (pd_yt_x * (
                                inner_grad[2] + (inner_grad[3] + inner_grad[0] + inner_grad[1].sum(1)).sum(
                            1))) * ((1 - j) / j) + W_grad[0] * (1 / j)

                student.conv1.bias.grad = grad_1[0]
                student.conv1.lin.weight.grad = grad_1[1]
                student.conv2.bias.grad = grad_1[2]
                student.conv2.lin.weight.grad = grad_1[3]
                optimizer_student.step()

            student.eval()
            p_s = student(data)[data.val_mask].max(1)[1]
            a_s = p_s.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
            if a_s > best_acc:
                best_acc = a_s
                torch.save(student.state_dict(), 'best_student')

            loss_s_c = F.cross_entropy(F.softmax(student(data), dim=1)[clean_val_mask], data.y[clean_val_mask])
            grad_inn = torch.autograd.grad(loss_s_c, student.parameters(), retain_graph=True)
            grad_inner = grad_inn[2] + (grad_inn[3] + grad_inn[0] + grad_inn[1].sum(1)).sum(1)
            Weight_matrix.weight.grad = grad_inner * pd_yt_x
            optimizer_Weight.step()

        student.eval()
        student.load_state_dict(torch.load('best_student'))
        student_acc = student(data)[data.test_mask].max(1)[1].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

        if iters == args.iteration - 1:
            print('Student_acc: ', student_acc)
            Student_ACC.append(torch.tensor(student_acc))

        if iters != args.iteration - 1:
            remove_select = select_noisy(args, data, student)
            '''移除噪声节点标签'''
            for i in remove_select:
                data.train_mask[i] = 0

            labels_idx_S, labels_idx_T, label_idex = select_pseudo(student(data), pre_add, data, args)

            for i in range(len(label_idex)):
                for j in label_idex[i]:
                    data.train_mask[j] = 1
                    data.y[j] = i


# 阻止SUGRL弹出警告
warnings.filterwarnings('ignore')
torch.cuda.set_device(0)
for dat in ['dblp']:  # 'Cora', 'CiteSeer', 'dblp', 'Photo', 'Computers'
    # 加载参数
    args = arg_parse(dat, noise_type='pair')  # 'uniform', 'pair'
    with open('utils/args.yaml', 'r') as file:
        parms_all = yaml.load(file, Loader=yaml.FullLoader)
    parms = parms_all[args.dataset_name]
    args.noise_seed = parms['noise_seed']
    for key in parms[args.noise_type].keys():
        setattr(args, key, parms[args.noise_type][key])

    '''获得多教师嵌入'''
    embeds_1 = get_embed1(args).to(args.device)
    embeds_2 = get_embed2(args).to(args.device)
    embeds_3 = get_embed3(args).to(args.device)

    sim_teacher1 = sim(embeds_1)
    sim_teacher2 = sim(embeds_2)
    sim_teacher3 = sim(embeds_3)
    sim_add = sim_teacher1 + sim_teacher2 + sim_teacher3

    for n_rate in [0.18, 0.19, 0.20, 0.21, 0.22, 0.38, 0.39, 0.40, 0.41, 0.42]:
        Student_ACC = []
        print("noise rate:", n_rate)
        for k in range(5):
            print("seed:", k)
            args.random_seed = k
            main(args, n_rate)

        Student_ACC = torch.stack(Student_ACC)
        print("Acc of Our: {:.4f}, std:{:.4f}, noise_rate: {}\n".format(Student_ACC.mean().item() * 100,
                                                                        Student_ACC.std().item() * 100,
                                                                        n_rate), Student_ACC)