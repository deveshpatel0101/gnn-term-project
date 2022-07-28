import torch
import numpy as np
import scipy.sparse as sp
import dgl


USER = 0


def load_data():

    file_edges = f'data/facebook/{USER}.edges'
    file_feat = f'data/facebook/{USER}.feat'

    edges_u, edges_v = [], []

    with open(file_edges) as f:
        for line in f:
            e1, e2 = tuple(int(x) - 1 for x in line.split())
            edges_u.append(e1)
            edges_v.append(e2)

    edges_u, edges_v = np.array(edges_u), np.array(edges_v)

    num_nodes = 0
    feats = []

    with open(file_feat) as f:
        for line in f:
            num_nodes += 1
            a = [int(x) for x in line.split()[1:]]
            feats.append(torch.tensor(a, dtype=torch.float))

    feats = torch.stack(feats)

    g = dgl.graph((edges_u, edges_v))
    g.ndata['feat'] = feats

    TEST_RATIO = 0.3

    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * TEST_RATIO)

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]
                                   ], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]
                                     ], neg_v[neg_eids[test_size:]]

    train_pos_g = dgl.graph((train_pos_u, train_pos_v),
                            num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v),
                            num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v),
                           num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v),
                           num_nodes=g.number_of_nodes())

    train_g = dgl.remove_edges(g, eids[:test_size])
    train_g = dgl.add_self_loop(train_g)

    return u, v, g, num_nodes, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g
