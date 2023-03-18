import math
from platform import node
from turtle import forward
from loguru import logger
from matplotlib import use
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import argparse
import numpy as np
from torch_geometric.utils import softmax
from torch_scatter import scatter, scatter_logsumexp


def _sim( z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        z = torch.mm(z1, z2.t())
        return torch.exp(z / 0.2)

def cos_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class ViewWeight(nn.Module):
    def __init__(self,nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(nhid,1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self,view1,view2):
        view1 = F.normalize(view1,p=2,dim=-1)
        view2 = F.normalize(view2,p=2,dim=-1)
        lamda1 =  torch.exp(torch.matmul(view1,self.weight))
        lamda2 =  torch.exp(torch.matmul(view2,self.weight))
        sum = lamda1 + lamda2 + 1e-8
        lamda_1 = lamda1 / sum
        lamda_2 = lamda2 / sum
        out = lamda_1 * view1 + lamda_2 * view2   
        return out

class MetricCalcLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        return h * self.weight

class MetricRelation(nn.Module):
    def __init__(self, nhid):
        super(MetricRelation).__init__()
        self.weight = nn.Parameter(torch.zeros(1,nhid))

    def forward(self, h):
        return  h + self.weight

class GraphGenerator(nn.Module):
    """
    Generate graph using similarity.
    """

    def __init__(self, dim, num_head=2, threshold=0.1, dev=None):
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(MetricCalcLayer(dim))
        self.num_head = num_head
        self.dev = dev

    def forward(self, left_h, right_h):
        """

        Args:
            left_h: left_node_num * hidden_dim/feat_dim
            right_h: right_node_num * hidden_dim/feat_dim
        Returns:

        """
        if torch.sum(left_h) == 0 or torch.sum(right_h) == 0:
            return torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        s = torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        zero_lines = torch.nonzero(torch.sum(left_h, 1) == 0)
        # The ReLU function will generate zero lines, which lead to the nan (devided by zero) problem.
        if len(zero_lines) > 0:
            left_h[zero_lines, :] += 1e-8
        for i in range(self.num_head):
            weighted_left_h = self.metric_layer[i](left_h)
            weighted_right_h = self.metric_layer[i](right_h)
            s += cos_sim(weighted_left_h, weighted_right_h)
        s /= self.num_head
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc1 = GraphConvolution(nfeat, nout)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout
        self.act = torch.tanh
        # self.layernorm = nn.LayerNorm(nhid)
    def forward(self, x, adj):
        # x = F.relu(self.gc1(x, adj))
        x = self.act(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.relu(self.gc1(x, adj))
        return x

class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.5)  # equal weight
        # if weights != None:
        #     # self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))
        #     with torch.no_grad():
        #         w = torch.Tensor(weights).reshape(self.weight.shape)
        #         self.weight.copy_(w)

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        # Row normalization of all graphs generated
        # adj_list = F.normalize(adj_list, dim=1, p=1)
        # Hadamard product + summation -> Conv
        return torch.sum(adj_list * F.tanh(self.weight), dim=0)

class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# GAT
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, att_features, dropout, alpha, concat=True,type="GAT"):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.type = type

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.W_mps = nn.Parameter(torch.empty(size=(att_features, out_features)))
        nn.init.xavier_uniform_(self.W_mps.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = nn.LayerNorm(out_features)
    # 
    def forward(self,h, adj, mps=None):
        h = torch.mm(h, self.W)   # [N, out_features]
        x = h[adj._indices()[0]]
        nei = h[adj._indices()[1]]    # 邻居节点的特征
        nei_x = torch.cat([x,nei],dim=-1)
        if mps is None:     
            e = self.leakyrelu(torch.matmul(nei_x,self.a))
        else:
            mps = torch.mm(mps, self.W_mps) 
            x_mps = mps[adj._indices()[0]]
            nei_mps = mps[adj._indices()[0]]
            nei_x_mps = torch.concat([x_mps,nei_mps],dim = -1)
            e = self.leakyrelu(torch.matmul(nei_x_mps,self.a))
        attention = softmax(e,adj._indices()[0])    #softmax 来自PYG
        attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        # nei_info = torch.mul(x,nei)
        # h_prime = attention * nei_x
        if self.type == "GAT":
            h_prime = attention * nei
        elif self.type == "edges":
            h_prime = attention * nei_x
        h_prime = scatter(h_prime,adj._indices()[0],dim=0,reduce="sum")
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime     
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat,nhid, nAtt, dropout, alpha, nheads,type="GAT"):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        # 多个图注意力层
        if type == "GAT":
            self.attentions = [GraphAttentionLayer(nfeat, int(nhid ), nAtt, dropout=dropout, alpha=alpha, concat=True,type=type) for _ in range(nheads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)
            # 输出层
            self.out_att = GraphAttentionLayer(nhid * nheads, int(nhid ), nhid * nheads, dropout=dropout, alpha=alpha, concat=False,type=type)
        elif type == "edges":
            self.attentions = [GraphAttentionLayer(nfeat, int(nhid / 2), nAtt, dropout=dropout, alpha=alpha, concat=True,type=type) for _ in range(nheads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)
            # 输出层
            self.out_att = GraphAttentionLayer(nhid * nheads, int(nhid / 2), nhid * nheads, dropout=dropout, alpha=alpha, concat=False,type=type)
        self.layernorm = nn.LayerNorm(nhid * nheads)

    def forward(self, x, adj, mps=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, mps) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.layernorm(x)
        x = F.elu(self.out_att(x, adj))
        return x
        # return F.log_softmax(x, dim=1)

class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=0)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
            
    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        t = beta.data.cpu().numpy()
        logger.info(f'Meta path attention : {t}')  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp 
    

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def _sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        z = torch.mm(z1, z2.t())
        return torch.exp(z / self.tau)

    def forward(self, z_mp, z_sc, pos):
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)

        matrix_mp2sc = self._sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()

        pos = pos.to_dense()

        matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1)).mean()
        
        matrix_sc2mp = matrix_sc2mp/(torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos).sum(dim=-1)).mean()
        
        return lori_mp + lori_sc

