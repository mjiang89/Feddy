"""
Author: Meng Jiang (mjiang2@nd.edu)
"""
import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import DGLGraph
from video_graph import VideoGraphDataset

from gcn import GCN
from gat import GAT
from sage import GraphSAGE

import copy
from torch.utils.data import DataLoader, Dataset


class CampusDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feats, label = self.data[item], self.targets[item]
        return feats, label

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        feats, label = self.dataset[self.idxs[item]]
        return feats, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, mask=None):
        self.args = args
        self.mask = mask
        self.loss_func = nn.MSELoss()
        self.ldr_train = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, model):
        model.train()
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        epoch_loss = []
        for local_epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (feats, labels) in enumerate(self.ldr_train):
                model.zero_grad()
                preds = model(feats)
                loss = self.loss_func(preds[self.mask], labels[self.mask])
                loss.backward()
                optimizer.step()
                '''
                if batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        local_epoch, batch_idx * len(feats), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                '''
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def load_data(args):
    data = VideoGraphDataset(args.dataset, args.tgraph, args.gpred, args.gtrain, args.gvalid, args.gtest)
    return data

def calc_error(preds, labels):
    errorX = torch.sqrt(torch.mean((preds[:, 0] - labels[:, 0]) ** 2))
    errorY = torch.sqrt(torch.mean((preds[:, 1] - labels[:, 1]) ** 2))
    return errorX, errorY

def eval_error(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        preds = model(features)
        preds = preds[mask]
        labels = labels[mask]
        errorX = torch.sqrt(torch.mean((preds[:, 0] - labels[:, 0]) ** 2))
        errorY = torch.sqrt(torch.mean((preds[:, 1] - labels[:, 1]) ** 2))
        return errorX, errorY

def iid_users(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def main(args):
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.FloatTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)

    g = data.graph 
    n_feats = features.shape[1]
    n_labels = data.num_labels
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
        #Features %d    
        #Edges %d
        #Labels %d
        #Train samples %d
        #Val samples %d
        #Test samples %d""" %
            (n_feats, n_edges, n_labels,
                train_mask.int().sum().item(),
                val_mask.int().sum().item(),
                test_mask.int().sum().item()))

    dataset_train = CampusDataset(features, labels)
    dict_users = iid_users(dataset_train, args.n_users)

    if args.gnnbase == 'gcn':
        g = DGLGraph(g)
        n_edges = g.number_of_edges()
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)
        model = GCN(g,
                n_feats,
                args.n_hidden,
                n_labels,
                args.n_layers,
                F.relu,
                args.dropout)

    if args.gnnbase == 'gat':
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
        g.add_edges(g.nodes(), g.nodes())
        n_edges = g.number_of_edges()
        heads = ([args.n_heads] * args.n_layers) + [args.n_out_heads]
        model = GAT(g,
                args.n_layers,
                n_feats,
                args.n_hidden,
                n_labels,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)

    if args.gnnbase == 'sage':
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
        n_edges = g.number_of_edges()
        model = GraphSAGE(g,
                n_feats,
                args.n_hidden,
                n_labels,
                args.n_layers,
                F.relu,
                args.dropout,
                args.aggregator_type)

    print(model)
    model.train()
    
    w_glob = model.state_dict()
    loss_train = []
    timecost = []

    for epoch in range(args.n_epochs):
        time_begin = time.time()

        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.n_users), 1)
        idxs_users = np.random.choice(range(args.n_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], mask=train_mask)
            w, loss = local.train(model=copy.deepcopy(model))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        w_glob = FedAvg(w_locals)

        model.load_state_dict(w_glob)

        time_end = time.time()
        timecost.append(time_end - time_begin)

        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Epoch {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)

        train_errX, train_errY = eval_error(model, features, labels, train_mask)
        val_errX, val_errY = eval_error(model, features, labels, val_mask)
        test_errX, test_errY = eval_error(model, features, labels, test_mask)        
        print("Epoch {:3d} | TrainRMSEX {:.4f} | TrainRMSEY {:.4f} | ValRMSEX {:.4f} | ValRMSEY {:.4f} | TestRMSEX {:.4f} | TestRMSEY {:.4f}"
                .format(epoch, train_errX, train_errY, val_errX, val_errY, test_errX, test_errY))

    print("Time cost {:.4f}".format(sum(timecost)/args.n_epochs))

    base_errX, base_errY = calc_error(features[test_mask,:2], labels[test_mask])
    print("TestRMSEX-Base {:.4f} | TestRMSEY-Base {:.4f}".format(base_errX, base_errY))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedGNN')
    parser.add_argument("--dataset", type=str, default="campb0", help="The input dataset. Can be [campb0], etc. " \
            +"Naming: [camp]...[b]ookstore[0-6], [c]oupa[0-3], [d]eathcircle[0-4], [g]ates[0-8], " \
            +"[h]yang[0-14], [l]ittle[0-3], [n]exus[0-11], [q]uad[0-3]")
    parser.add_argument("--tgraph", type=int, default=30, help="number of video frames between graphs")
    parser.add_argument("--gpred", type=int, default=5, help="number of graphs after to predict")
    parser.add_argument("--gtrain", type=int, default=60, help="number of graphs for training")
    parser.add_argument("--gvalid", type=int, default=60, help="number of graphs for validation")
    parser.add_argument("--gtest", type=int, default=180, help="number of graphs for test")

    # model choice
    parser.add_argument("--gnnbase", type=str, default="gcn", help="gnn base model: [gcn], [gat], [sage]")

    # gcn & gat & sage
    parser.add_argument("--n_epochs", type=int, default=30, help="number of training epochs")
    parser.add_argument("--n_layers", type=int, default=1, help="number of hidden layers")
    parser.add_argument("--n_hidden", type=int, default=512, help="number of hidden units")
    parser.add_argument("--lr", type=float, default=.01, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")

    # gcn & sage
    parser.add_argument("--dropout", type=float, default=.5, help="dropout probability")

    # gat
    parser.add_argument("--n_heads", type=int, default=8, help="number of hidden attention heads")
    parser.add_argument("--n_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.6, help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.6, help="attention dropout")
    parser.add_argument('--negative_slope', type=float, default=.2, help="negative slope of leaky relu")

    # sage
    parser.add_argument("--aggregator_type", type=str, default="gcn", help="aggregator type: mean/gcn/pool/lstm")

    # feddy
    parser.add_argument("--n_users", type=int, default=10, help="number of users")
    parser.add_argument('--frac', type=float, default=.1, help="fraction of clients")
    parser.add_argument('--local_ep', type=int, default=100, help="number of local epochs")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size")

    args = parser.parse_args()
    print(args)

    main(args)

