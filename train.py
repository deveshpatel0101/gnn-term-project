from sklearn.metrics import roc_auc_score
from models.AGNN import AGNN
from models.Cheb import Cheb
from models.GAT import GAT
from models.GCN import GCN
from models.GraphSAGE import GraphSAGE
from link_prediction.dot_predictor import DotPredictor
import torch
import itertools
import torch.nn.functional as F


def pipeline(train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, model_name='GCN', hidden_size=16, epochs=100, lr=0.01):
    if model_name == 'GCN':
        model = GCN(train_g.ndata['feat'].shape[1], hidden_size)
    elif model_name == 'SAGE':
        model = GraphSAGE(train_g.ndata['feat'].shape[1], hidden_size)
    elif model_name == 'GAT':
        model = GAT(train_g, train_g.ndata['feat'].shape[1], hidden_size)
    elif model_name == 'CHEB':
        model = Cheb(train_g.ndata['feat'].shape[1], hidden_size)
    elif model_name == 'AGNN':
        model = AGNN(train_g.ndata['feat'].shape[1], hidden_size)

    pred = DotPredictor()

    optimizer = torch.optim.Adam(itertools.chain(
        model.parameters(), pred.parameters()), lr)

    train_ac, test_ac = [], []
    train_loss, test_loss = [], []
    for e in range(epochs):
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(f'------ Epoch {e}/{epochs} ------')
            train_ac.append(get_accuracy(train_pos_g, train_neg_g, pred, h))
            test_ac.append(get_accuracy(test_pos_g, test_neg_g, pred, h))
            train_loss.append(loss.item())
            test_loss.append(compute_loss(
                pred(test_pos_g, h), pred(test_neg_g, h)).item())

    print('Train acc', train_ac[-1], f'| Train loss: {train_loss[-1]}')
    print('Test acc', test_ac[-1], f'| Test loss: {test_loss[-1]}')
    print(train_ac, test_ac, train_loss, test_loss)
    return h


def get_accuracy(pos_g, neg_g, pred, h):
    with torch.no_grad():
        pos_score = pred(pos_g, h)
        neg_score = pred(neg_g, h)
    return compute_auc(pos_score, neg_score)


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]),
                        torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)
