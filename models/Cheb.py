from dgl.nn import ChebConv
import torch.nn as nn
import torch.nn.functional as F


class Cheb(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(Cheb, self).__init__()
        self.conv1 = ChebConv(in_feats, h_feats, 8)
        self.conv2 = ChebConv(h_feats, h_feats, 8)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
