from dgl.nn import AGNNConv
import torch.nn as nn
import torch.nn.functional as F


class AGNN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(AGNN, self).__init__()
        self.conv1 = AGNNConv(in_feats, h_feats)
        self.conv2 = AGNNConv(h_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
