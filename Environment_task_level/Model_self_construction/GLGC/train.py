from network import Network
from metric import valid
from torch.utils.data import Dataset
import argparse
from loss import Loss
from dataloader import load_data
from util import *




Dataname = 'DHA'
# Dataname = 'Land'
# Dataname = 'ProteinFold'
# Dataname = 'ALOI'

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--choice', default="mask", type=str)  # mask noise NI
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=5)
parser.add_argument("--alpha", default=0.1)
parser.add_argument("--beta", default=1)
parser.add_argument("--eta", default=0.2)
parser.add_argument("--pos", default=0.1)
parser.add_argument("--neg", default=0.2)
parser.add_argument("--miss_rate", default=0.5)  # 0.1 0.3 0.5 0.7 1.0
parser.add_argument("--weight", default=True)
args = parser.parse_args()
print(args.choice)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(3)
# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.

if args.dataset == "DHA":
    seed = 9
    args.high_feature_dim = 5
    if args.choice in ["mask", "NI"]:
        args.con_epochs = 100
        args.pos = 0.3
        args.neg = 0.3
    if args.choice == "noise":
        args.con_epochs = 150
        args.pos = 0.2
        args.neg = 0.3
if args.dataset == "ProteinFold":
    args.con_epochs = 200
    seed = 10
    args.high_feature_dim = 25
    if args.choice in ["mask", "NI"]:
        args.pos = 0.1
        args.neg = 0.2
    if args.choice == "noise":
        args.pos = 0.1
        args.neg = 0.1
if args.dataset == "Land":
    args.con_epochs = 150
    seed = 10
    args.high_feature_dim = 15
    if args.choice in ["mask", "NI"]:
        args.pos = 0.2
        args.neg = 0.3
    if args.choice == "noise":
        args.pos = 0.2
        args.neg = 0.3
if args.dataset == "ALOI":
    args.con_epochs = 200
    seed = 10
    args.high_feature_dim = 10
    if args.choice in ["mask", "NI"]:
        args.pos = 0.3
        args.neg = 0.3
    if args.choice == "noise":
        args.pos = 0.2
        args.neg = 0.4



def setup_seed(seed):
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def pretrain(epoch):
    tot_loss = 0.
    if args.choice in ["mask", "NI"]:
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        criterion = torch.nn.MSELoss()
    # criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        com_mask = mask[idx]
        if isinstance(com_mask, np.ndarray):
            com_mask = torch.from_numpy(com_mask)
        com_mask = com_mask.float().to(device)
        optimizer.zero_grad()
        _, xrs, _ = model(xs)
        loss_list = []
        if args.choice in ["mask", "NI"]:
            for v in range(view):
                # Calculate the reconstruction loss for each sample，shape: [batch_size]
                rec_loss = criterion(xs[v], xrs[v]).mean(dim=1)

                # Retrieve the valid samples of the current batch in this view via mask
                valid_mask = com_mask[:, v]  # [batch_size]

                # Loss of retaining only valid samples
                masked_loss = rec_loss * valid_mask

                # Calculate the average loss of the view (only considering valid samples)
                if valid_mask.sum() > 0:
                    loss_v = masked_loss.sum() / (valid_mask.sum())
                    loss_list.append(loss_v)

        else:
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v]))

        # At least one view must have valid samples for backpropagation
        if loss_list:
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))



def contrastive_train(epoch):
    tot_loss = 0.
    if args.choice in ["mask", "NI"]:
        mse = torch.nn.MSELoss(reduction='none')
    else:
        mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, xrs, zs = model(xs)
        com_mask = mask[idx]
        if isinstance(com_mask, np.ndarray):
            com_mask = torch.from_numpy(com_mask)
        com_mask = com_mask.float().to(device)
        # local graph
        G_intra, G_inter = robust_affinity(hs, com_mask, args)
        loss_list = []

        if args.choice in ["mask", "NI"]:
            # ggc loss
            loss_list.append(args.alpha * criterion.forward_intre_mask(hs, com_mask))
            for v in range(view):
                for w in range(v + 1, view):
                    # lwc loss
                    loss_list.append(args.beta * criterion.forward_inter_mask(hs[v], hs[w], com_mask[:, [v, w]], G_inter[v,w], G_inter[w,v]))

                # Calculate the reconstruction loss for each sample，shape: [batch_size]
                rec_loss = mse(xs[v], xrs[v]).mean(dim=1)
                # Retrieve the valid samples of the current batch in this view via mask
                valid_mask = com_mask[:, v]  # [batch_size]
                # Loss of retaining only valid samples
                masked_loss = rec_loss * valid_mask
                # Calculate the average loss of the view (only considering valid samples)
                if valid_mask.sum() > 0:
                    loss_v = masked_loss.sum() / (valid_mask.sum())
                    loss_list.append(loss_v)

        else:
            # ggc loss
            loss_list.append(args.alpha * criterion.forward_intre_noise(hs))
            for v in range(view):
                for w in range(v+1, view):
                    # lwc loss
                    loss_list.append(args.beta * criterion.forward_inter_noise(hs[v], hs[w], G_inter[v,w], G_inter[w,v]))
                loss_list.append(mse(xs[v], xrs[v]))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))



T = 5
accs = []
nmis = []
aris = []
# if not os.path.exists('models'):
#     os.makedirs('models')
for i in range(T):

    dataset, dims, view, data_size, class_num = load_data(args.dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    print("ROUND:{}".format(i+1))
    setup_seed(i+seed)
    mask = get_mask(view, data_size, args.miss_rate)

    # If executing the NI task, cancel the comments on the following four lines
    # add_gaussian_noise(dataset, mask)
    # setup_seed(i+seed+1)
    # mask = get_mask(view, data_size, args.miss_rate)
    # setup_seed(i + seed)

    if args.choice == "noise" and args.miss_rate > 0.:
        add_gaussian_noise(dataset, mask)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, args.pos, args.neg, args.weight, device).to(device)

    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_train(epoch)
        if epoch == args.mse_epochs + args.con_epochs:
            acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num, mask, args)
            if epoch == args.mse_epochs + args.con_epochs:
                accs.append(acc)
                nmis.append(nmi)
                aris.append(ari)
        epoch += 1

print(accs, f"{np.mean(accs)*100:.2f}", f"{np.std(accs)*100:.1f}")
print(nmis, f"{np.mean(nmis)*100:.2f}", f"{np.std(nmis)*100:.1f}")
print(aris, f"{np.mean(aris)*100:.2f}", f"{np.std(aris)*100:.1f}")
