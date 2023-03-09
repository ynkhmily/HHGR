from loguru import logger
from matplotlib import use
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from base import HypergraphConv
from layers import Contrast,Attention


class GCN_RE(nn.Module):
    def __init__(self, args,):
        super(GCN_RE, self).__init__()
        # ! Init variables
        self.device = args.device
        self.hidden_dim = args.hidden_dim
        self.num_heads = args.nheads
        self.out_dim = args.out_dim

        self.fc_list = nn.ModuleList([
            nn.Sequential(
            nn.Linear(feats_dim, feats_dim * 2),
            nn.ELU(),
            nn.Linear(feats_dim * 2, self.hidden_dim)
        ) for feats_dim in args.in_dims])

        
        self.feat_drop = nn.Dropout(args.dropout)
        self.num_metapaths = args.num_metapaths


        self.hyperConv = HypergraphConv(self.hidden_dim,args.out_dim,num_class=len(args.in_dims),use_attention=True,heads=args.nheads)
       
        self.contrast = Contrast(args.out_dim,0.8,0.5)
        
        self.Attention_1 = Attention(args.out_dim,0.2)

        self.tanh = torch.tanh

    def forward(self, inputs):
        hyperedge, metapath_adjs, features_list, type_mask = inputs
        
        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = torch.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        x = self.feat_drop(transformed_features)
        

        x1 = self.tanh(self.hyperConv(x, hyperedge[0],type_mask,
                                      hyperedge_attr=x[type_mask == 0]))[type_mask == 0]

        
        x2 = [self.tanh(self.hyperConv(x[type_mask == 0], adj)) for adj in metapath_adjs]
        

        return x1, x2

    def loss(self, inputs, pos):      
        
        x1, x2 = self.forward(inputs)

        loss = 0
        
        x2.append(x1)
        p2 = self.Attention_1(x2)
        
        for i in range(self.num_metapaths + 1):
            loss += self.contrast(p2,x2[i],pos)
            for j in range(i,self.num_metapaths + 1):
                loss += self.contrast(x2[i],x2[j],pos) 
 
        
        return loss

    def get_embeds(self, inputs):
        x1, x2 = self.forward(inputs)
        
        x2.append(x1)
        p2 = self.Attention_1(x2)
        
        out = p2
        
        return out.detach()
