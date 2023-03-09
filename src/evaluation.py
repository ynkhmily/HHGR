import numpy as np
import torch
from logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
from loguru import logger
from sklearn.metrics import roc_auc_score, adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from tqdm import tqdm
from torch.optim import Adam
from munkres import Munkres
##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################


seed = 2022 #DBLP 
# seed = 2022 #DBLP 
import os
import random
import torch_geometric
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
torch_geometric.seed_everything(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def evaluate(embeds, ratio, idx_train, idx_val, idx_test, label, device, dataset, lr, wd
             , isTest=True):
    hid_units = embeds.shape[1]
    
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]


    train_lbls = label[idx_train]
    val_lbls = label[idx_val]
    test_lbls = label[idx_test]
    nb_classes = torch.max(label).item() + 1

    
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))

    if isTest:
        logger.info("[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc-mean {:.4f} var: {:.4f}",
                    np.mean(macro_f1s),
                    np.std(macro_f1s),
                    np.mean(micro_f1s),
                    np.std(micro_f1s),
                    np.mean(auc_score_list),
                    np.std(auc_score_list)
                    )
              
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)


def find_epoch(hid_units, nb_classes, train_embs, train_lbls, test_embs, test_lbls,device):
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.001, weight_decay=0.00001)
    xent = nn.CrossEntropyLoss()
    log.to(device)

    epoch_flag = 0
    epoch_win = 0
    best_acc = torch.zeros(1).to(device)

    for e in range(20000):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward(retain_graph=True)
        opt.step()

        if (e + 1) % 10 == 0:
            log.eval()
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            if acc >= best_acc:
                epoch_flag = e + 1
                best_acc = acc
                epoch_win = 0
            else:
                epoch_win += 1
            if epoch_win == 10:
                break
    return epoch_flag

def node_classification(embeds, train_node, valid_node, test_node, train_target, valid_target, test_target,device):
    train_embeds = embeds[train_node, :]
    valid_embeds = embeds[valid_node, :]
    test_embeds = embeds[test_node, :]

    num_class = torch.max(train_target).item() + 1
    
    # node_dim = args.n_hidden2
    node_dim = embeds.shape[1]
    
    xent = nn.CrossEntropyLoss()

    log = LogReg(node_dim, num_class)
    log.to(device)
    logger.info('Searching for property number of epoch...')
    n_of_log = find_epoch(node_dim, num_class, train_embeds, train_target, test_embeds, test_target,device)
    logger.info('Node Classify Epoches: ', n_of_log)
    logger.info('Classifing now...')
    opt = Adam(log.parameters(), lr=0.001, weight_decay=0.00001)
    for _ in tqdm(range(n_of_log)):
        log.train()
        opt.zero_grad()
        logits = log(train_embeds)
        cls_loss = xent(logits, train_target)
        cls_loss.backward(retain_graph=True)
        opt.step()
    logits = log(test_embeds)
    preds = torch.argmax(logits, dim=1)
    preds, test_target = preds.cpu().numpy(), test_target.cpu().numpy()
    micro_f1_test = f1_score(preds, test_target, average='micro') * 100
    macro_f1_test = f1_score(preds, test_target, average='macro') * 100    
    logger.info('Test set index:')
    logger.info('Micro F1: {:.3f}%, Macro F1: {:.3f}%'.format(micro_f1_test, macro_f1_test))

    logits = log(train_embeds)
    preds = torch.argmax(logits, dim=1)
    preds, train_target = preds.cpu().numpy(), train_target.cpu().numpy()
    micro_f1 = f1_score(preds, train_target, average='micro') * 100
    macro_f1 = f1_score(preds, train_target, average='macro') * 100
    logger.info('Train set index:')
    logger.info('Micro F1: {:.3f}%, Macro F1: {:.3f}%'.format(micro_f1, macro_f1))

    logits = log(valid_embeds)
    preds = torch.argmax(logits, dim=1)
    preds, valid_target = preds.cpu().numpy(), valid_target.cpu().numpy()
    micro_f1 = f1_score(preds, valid_target, average='micro') * 100
    macro_f1 = f1_score(preds, valid_target, average='macro') * 100
    logger.info('Valid set index:')
    logger.info('Micro F1: {:.3f}%, Macro F1: {:.3f}%'.format(micro_f1, macro_f1))

    return micro_f1_test, macro_f1_test

def node_cluster(embeds, labels):       # 对所得点的表示进行聚类，默认算法是kmeans
    num_nodes = labels.shape[0]
    embeds, labels = embeds[:num_nodes].detach().cpu().numpy(), labels.detach().cpu().numpy()
    from sklearn.cluster import KMeans
    logger.info('Clustering nodes with algorithm KMeans...')
    num_class = np.max(labels).item() + 1
    kmeans = KMeans(n_clusters=num_class, random_state=0).fit(embeds)
    preds = kmeans.predict(embeds)
    preds = fix_preds(preds, labels)
    
    accuracy = accuracy_score(labels, preds) * 100
    micro_f1 = f1_score(labels, preds, average='micro') * 100
    macro_f1 = f1_score(labels, preds, average='macro') * 100
    ARI = adjusted_rand_score(labels, preds) * 100
    NMI = normalized_mutual_info_score(labels, preds) * 100
    logger.info('Acc: {:.3f}%, Micro F1: {:.3f}%, Macro F1: {:.3f}%, ARI: {:.3f}%, NMI: {:.3f}%'.format(accuracy, micro_f1, macro_f1, ARI, NMI))
    return accuracy, micro_f1, macro_f1, ARI, NMI       # 聚类算法的五个指标


def fix_preds(preds, labels):       # 对聚类所得的标签进行修正
    
    m = Munkres()
    num_class = np.max(labels).item() + 1
    # label_type_1 = list(set(labels))
    label_type_1 = list(np.unique(labels))
    label_type_2 = list(np.unique(preds))
    # label_type_2 = list(set(preds))
    cost = np.zeros((num_class, num_class), dtype=int)
    for i, c1 in enumerate(label_type_1):
        mps = [i1 for i1, e1 in enumerate(labels) if e1 == c1]
        for j, c2 in enumerate(label_type_2):
            mps_d = [i1 for i1 in mps if preds[i1] == c2]
            cost[i][j] = len(mps_d)
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    actual_preds = np.zeros(preds.size, dtype=int)
    for i, c in enumerate(label_type_1):
        c2 = label_type_2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(preds) if elm == c2]
        actual_preds[ai] = c
    return actual_preds