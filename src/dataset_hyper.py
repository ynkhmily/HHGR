from curses.ascii import islower
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pickle 
import torch
import random
import torch.nn.functional as F
import torch_geometric


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)).astype(float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def load_ACM_data(prefix='./data/ACM',ratio=[20,40,60]):
    pap_hyperedge = np.load("./hyperedge/ACM/ACM_PAP_hyperedge.npy")
    psp_hyperedge = np.load("./hyperedge/ACM/ACM_PSP_hyperedge.npy")
    pap_hyperedge = torch.LongTensor(pap_hyperedge)
    psp_hyperedge = torch.LongTensor(psp_hyperedge)

    hyperedge = np.load("./hyperedge/ACM/ACM_hyperedge.npy")
    hyperedge = torch.LongTensor(hyperedge)

    features_0 = sp.load_npz(prefix + '/p_feat.npz').toarray()
    features_1 = sp.eye(7167,dtype=np.float32).toarray()
    features_2 = sp.eye(60,dtype=np.float32).toarray()

    labels = np.load(prefix + '/labels.npy')
    type_mask = np.load(prefix + '/node_types.npy')

    pos = sp.load_npz(prefix + "/pos.npz")    
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    
    return [hyperedge], \
           [pap_hyperedge,psp_hyperedge],\
           [features_0,features_1,features_2],\
           type_mask,\
           labels, \
           pos


def load_DBLP_data(prefix='./data/DBLP_processed',ratio=[20,40,60]):
    apa_hyperedge = np.load("./hyperedge/DBLP/DBLP_APA_hyperedge.npy")
    apcpa_hyperedge = np.load("./hyperedge/DBLP/DBLP_APCPA_hyperedge.npy")
    aptpa_hyperedge = np.load("./hyperedge/DBLP/DBLP_APTPA_hyperedge.npy")
    apa_hyperedge = torch.LongTensor(apa_hyperedge)
    apcpa_hyperedge = torch.LongTensor(apcpa_hyperedge)
    aptpa_hyperedge = torch.LongTensor(aptpa_hyperedge)


    hyperedge = np.load("./hyperedge/DBLP/DBLP_hyperedge.npy")
    hyperedge = torch.LongTensor(hyperedge)


    features_0 = sp.load_npz(prefix + '/features_0.npz').toarray()
    #one-hot
    features_1 = np.eye(14328,dtype=np.float32)
    features_2 = np.eye(7723,dtype=np.float32)
    features_3 = np.eye(20, dtype=np.float32)


    labels = np.load(prefix + '/labels.npy')
    type_mask = np.load(prefix + '/node_types.npy')

    pos = sp.load_npz(prefix + "/pos.npz")    
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    
    return [hyperedge], \
           [apa_hyperedge,apcpa_hyperedge,aptpa_hyperedge],\
           [features_0,features_1,features_2, features_3],\
           type_mask,\
           labels, \
           pos
  

def load_Yelp_data(prefix='./data/yelp',ratio=[20,40,60]):
    bub_hyperedge = np.load("./hyperedge/Yelp/Yelp_BUB_hyperedge.npy")
    bsb_hyperedge = np.load("./hyperedge/Yelp/Yelp_BSB_hyperedge.npy")
    blb_hyperedge = np.load("./hyperedge/Yelp/Yelp_BLB_hyperedge.npy")
    bub_hyperedge = torch.LongTensor(bub_hyperedge)
    bsb_hyperedge = torch.LongTensor(bsb_hyperedge)
    blb_hyperedge = torch.LongTensor(blb_hyperedge)

    hyperedge = np.load("./hyperedge/Yelp/Yelp_hyperedge.npy")
    hyperedge = torch.LongTensor(hyperedge)

    with open(f'{prefix}/node_features.pkl', 'rb') as f:
        features = pickle.load(f)   

    features_0 = features[0:2614]
    features_1 = np.eye(1286,dtype=np.float32)
    features_2 = np.eye(4,dtype=np.float32)
    features_3 = np.eye(9,dtype=np.float32)


    labels = np.load(prefix + '/labels.npy')
    type_mask = np.load(prefix + '/0/train_mask.npy')

    pos = sp.load_npz(prefix + "/pos.npz")    
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    
    return [hyperedge], \
           [bub_hyperedge,bsb_hyperedge,blb_hyperedge],\
           [features_0,features_1,features_2, features_3],\
           type_mask,\
           labels, \
           pos


if __name__=="__main__":
    # load_YELP_data()
    # load_DBLP_data()
    # load_ACM_data()
    print()