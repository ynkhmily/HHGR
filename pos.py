import torch as th
import numpy as np
import pickle
import scipy.sparse as sp
import torch
import sys
def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def cos_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def count(all,pos_num,p):
    total = 0
    ones = th.ones_like(th.LongTensor(all))
    pos = np.zeros((p,p))
    k=0
    for i in range(len(all)):
        one = all[i].nonzero()[0]
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])
            sele = one[oo[:pos_num]]
            pos[i, sele] = 1
            total += pos_num
        else:
            pos[i, one] = 1
            total += len(one)
    
    row, col = np.diag_indices_from(pos)
    pos[row,col] = 1
    
    pos = sp.coo_matrix(pos)
    return pos, total

def get_sim_2(feat,k):
    feat = torch.FloatTensor(feat).cuda()
    matrix_sim = cos_sim(feat,feat)
    values, indices = torch.topk(matrix_sim,k)
    all = indices.cpu().numpy()
    total = 0
    p = feat.shape[0]
    pos = np.zeros((p,p))
    for i in range(len(all)):
        one = all[i]
        total += len(one)
        pos[i,one] = 1

    row, col = np.diag_indices_from(pos)
    pos[row,col] = 1
    
    pos = sp.coo_matrix(pos)
    return pos, total

def combine(pos1,pos2):
    pos = np.zeros((pos1.shape[0],pos1.shape[1]))
    total = 0
    for i,x in enumerate(pos1):
        y = pos2[i]
        x_nei = x.nonzero()[1]
        y_nei = y.nonzero()[1]
        # z_nei = np.intersect1d(x_nei,y_nei)
        z_nei = np.union1d(x_nei,y_nei)
        pos[i, z_nei] = 1
        total += len(z_nei)
    
    print("final-based total nei:",total)
    row, col = np.diag_indices_from(pos)
    pos[row,col] = 1
    
    pos = sp.coo_matrix(pos)
    return pos            

def yelp(k1,k2):
    labels = np.load("./data/yelp/labels.npy")
    pos_num = 3
    p = 2614

    with open(f'./data/yelp/node_features.pkl', 'rb') as f:
       feat = pickle.load(f)[:2614]

    bub = np.load("./data/yelp/bub.npy")
    bsb = np.load("./data/yelp/bsb.npy")
    blb = np.load("./data/yelp/blb.npy")

    all = (bub + bsb + blb)
    all_ = (all>0).sum(-1).astype("float32")
    
    # pos1, total = count(all,pos_num,p)
    mps = np.load("./data/yelp/metapath2vec_yelp.npy")[:2614]
    pos1, total = get_sim_2(mps,k1)
    print("mps-based total nei: ", total)

    # pos2, total = get_sim(all,feat,alpha=0.90)
    pos2, total = get_sim_2(feat,k2)
    print("raw-based total nei: ", total)
    # sp.save_npz("./data/yelp/pos.npz", pos)
    
    return pos1, pos2, labels

def acm(k1,k2):
    labels = np.load("./data/ACM/labels.npy")
    pos_num = 6
    p = 4019

    feat = sp.load_npz("./data/ACM/p_feat.npz").toarray()

    pap = sp.load_npz("./data/ACM/pap.npz").toarray()
    psp = sp.load_npz("./data/ACM/psp.npz").toarray()

    all = (pap + psp)
    all_ = (all>0).sum(-1).astype("float32")
    
    mps = np.load("./data/ACM/metapath2vec_acm.npy")[:p]
    pos1, total = get_sim_2(mps,k1)
    print("mps-based total nei: ", total)

    # pos2, total = get_sim(all,feat,alpha=0.90)
    pos2, total = get_sim_2(feat,k2)
    print("raw-based total nei: ", total)
    
    return pos1, pos2, labels

def dblp(k1,k2):
    labels = np.load("./data/DBLP_processed/labels.npy")
    p = 4057

    feat = sp.load_npz('./data/DBLP_processed/features_0.npz').toarray()[:4057]

    apa = sp.load_npz("./data/DBLP_processed/apa.npz").toarray()
    apa = np.array(apa,int)
    apcpa = sp.load_npz("./data/DBLP_processed/apcpa.npz").toarray()
    apcpa = np.array(apcpa,int)
    aptpa = sp.load_npz("./data/DBLP_processed/aptpa.npz").toarray()
    aptpa = np.array(aptpa,int)

    all = (apa + apcpa + aptpa)
    all_ = (all>0).sum(-1).astype("float32")
    
    mps = np.load("./data/DBLP_processed/metapath2vec_dblp.npy")[:p]
    pos1, total = get_sim_2(mps,k1)
    print("mps-based total nei: ", total)

    # pos2, total = get_sim(all,feat,alpha=0.90)
    pos2, total = get_sim_2(feat,k2)
    print("raw-based total nei: ", total)

    # sp.save_npz("./data/DBLP_processed/pos.npz", pos)
    return pos1, pos2, labels


# k1 = int(sys.argv[1])
# k2 = int(sys.argv[2])
# dataset = sys.argv[3]
k1 = 2
k2 = 4
dataset = "ACM"
print("dataset : {} generate pos! k1 : {}, k2 : {}".format(dataset,k1,k2))
# dataset = "dblp"
if dataset == "Yelp":
    pos1, pos2, labels = yelp(k1,k2)
elif dataset == 'DBLP':
    pos1, pos2, labels = dblp(k1,k2)
elif dataset == 'ACM':
    pos1, pos2, labels = acm(k1,k2)

pos1 = pos1.todense()
pos2 = pos2.todense()

pos = combine(pos1,pos2)

if dataset == "Yelp":
    sp.save_npz("./data/yelp/pos.npz",pos)
elif dataset == "ACM":
    sp.save_npz("./data/ACM/pos.npz",pos)
elif dataset == "DBLP":
    sp.save_npz("./data/DBLP_processed/pos.npz",pos)
