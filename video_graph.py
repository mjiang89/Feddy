"""
Author: Meng Jiang (mjiang2@nd.edu)
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp

class VideoGraphDataset(object):
    """
    Campus video graph dataset. Nodes mean pedestrians, bikers, vehicles, etc.
    and edges mean either different objects in the same graph
    or the same object in neighboring graphs.
    """
    def __init__(self, name, tgraph=30, gpred=5, gtrain=60, gvalid=60, gtest=180):
        """
        name => [b]ookstore[0-6], [c]oupa[0-3], [d]eathcircle[0-4], [g]ates[0-8],
            [h]yang[0-14], [l]ittle[0-3], [n]exus[0-11], [q]uad[0-3]
        tgraph => the number of video frames between graphs (if fps=30 as default, [30] means 1 second between graphs)
        gpred => the number of graphs after to predict ([5] means the objective is to predict
            position of objects after 5 graphs; if tgraph=30, [5] means it is to predict the future in 5 seconds)
        gtrain => the number of graphs for training (if tgraph=30, [60] means training with 1-minute data)
        gvalid => the number of graphs for validation (if tgraph=30, [60] means developing with 1-minute data)
        gtest => the number of graphs for test (if tgraph=30, [180] means testing on 3-minutes data)
        """
        self.name = name
        self.dir = 'data'
        self.tgraph = tgraph
        self.gpred = gpred
        self.gtrain = gtrain
        self.gvalid = gvalid
        self.gtest = gtest
        self._load()

    def _load(self):
        gmagic = 2

        data = []
        fr = open("{}/{}.txt".format(self.dir, self.name), 'r')
        fr.readline()
        for line in fr:
            arr = line.strip('\r\n').split(' ')
            node = []
            for i in range(3): node.append(int(arr[i]))
            for i in range(3,22): node.append(float(arr[i]))
            data.append(node)
        fr.close()

        tmmap = {}
        for node in data:
            tm = node[0]
            if not tm in tmmap: tmmap[tm] = len(tmmap)

        _data_ = []
        for node in data:
            tm = tmmap[node[0]]
            if tm % self.tgraph == 0:
                _data_.append(node)
        data = _data_
        n_nodes = len(data) # number of nodes

        tmmap, idxmap, tpmap = {}, {}, {}
        for node in data:
            tm,idx,tp = node[0],node[1],node[2]
            if not tm in tmmap: tmmap[tm] = len(tmmap)
            if not idx in idxmap: idxmap[idx] = len(idxmap)
            if not tp in tpmap: tpmap[tp] = len(tpmap)
        n_tps = len(tpmap)

        for i in range(n_nodes):
            node = data[i]
            node[0] = tmmap[node[0]]
            node[1] = idxmap[node[1]]
            tp = tpmap[node[2]]
            for j in range(n_tps):
                if j == tp:
                    node.append(1.)
                else:
                    node.append(0.)
            data[i] = node

        tm2idx2nodei = {}
        idx2tm2nodei = {}
        for i in range(n_nodes):
            node = data[i]            
            tm, idx = node[0], node[1]
            if not tm in tm2idx2nodei: tm2idx2nodei[tm] = {}
            tm2idx2nodei[tm][idx] = i
            if not idx in idx2tm2nodei: idx2tm2nodei[idx] = {}
            idx2tm2nodei[idx][tm] = i

        edges = []
        """ connecting objects in the same graph """
        '''
        for tm,idx2nodei in tm2idx2nodei.items():
            idxs = list(idx2nodei.keys())
            n_idxs = len(idxs)
            if n_idxs > 1:
                for i in range(n_idxs):
                    for j in range(n_idxs):
                        if i == j: continue
                        nodei = idx2nodei[idxs[i]]
                        nodej = idx2nodei[idxs[j]]
                        edges.append([nodei,nodej])
        '''
        """ connecting the same object in the neighboring graphs """
        for idx,tm2nodei in idx2tm2nodei.items():
            for tm,nodei in tm2nodei.items():
                if tm + gmagic in tm2nodei:
                    nodej = tm2nodei[tm + gmagic]
                    edges.append([nodej,nodei])
        n_edges = len(edges) # number of edges

        edges = np.asarray(edges, dtype=np.int32).reshape(n_edges, 2)
        adj = sp.coo_matrix((np.ones(n_edges), (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes), dtype=np.float32)
        self.graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())

        features, labels = [], []
        for i in range(n_nodes):
            node = data[i]
            tm, idx = node[0], node[1]
            features.append(node[3:-2])
            if idx in idx2tm2nodei and tm + self.gpred in idx2tm2nodei[idx]:
                nodei = idx2tm2nodei[idx][tm + self.gpred]
                labels.append([data[nodei][3], data[nodei][4]])
            else:
                labels.append([-1, -1])
        n_feats = len(features[0]) # number of features

        features = np.asarray(features, dtype=np.float32).reshape(n_nodes, n_feats)
        labels = np.asarray(labels, dtype=np.float32).reshape(n_nodes, 2)
        self.features = features
        self.labels = labels
        self.num_labels = 2

        train_mask = np.zeros(n_nodes)
        val_mask = np.zeros(n_nodes)
        test_mask = np.zeros(n_nodes)

        lasttm = data[-1][0]
        for i in range(n_nodes):
            if labels[i][0] < 0: continue
            node = data[i]
            tm = node[0]
            if tm >= self.gpred and tm < self.gpred + self.gtrain:
                train_mask[i] = 1
            elif tm > lasttm - self.gtest:
                test_mask[i] = 1
            elif tm > lasttm - self.gtest - self.gvalid:
                val_mask[i] = 1

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
