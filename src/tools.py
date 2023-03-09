import torch
import dgl
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from loguru import logger
def test(embeddings, labels, train_split=0.2, runs=10):
    macro_f1_list = list()
    micro_f1_list = list()
    nmi_list = list()
    ari_list = list()

    for i in range(runs):
        x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, train_size=train_split, random_state=i)

        clf = SVC(probability=True)

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        macro_f1 = f1_score(y_test, y_pred, average='macro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1_list.append(macro_f1)
        micro_f1_list.append(micro_f1)

    for i in range(runs):
        kmeans = KMeans(n_clusters=len(torch.unique(labels)), algorithm='full')
        y_kmeans = kmeans.fit_predict(embeddings)

        nmi = normalized_mutual_info_score(labels, y_kmeans)
        ari = adjusted_rand_score(labels, y_kmeans)
        nmi_list.append(nmi)
        ari_list.append(ari)

    macro_f1 = np.array(macro_f1_list).mean()
    micro_f1 = np.array(micro_f1_list).mean()
    nmi = np.array(nmi_list).mean()
    ari = np.array(ari_list).mean()

    logger.info("micro_f1: {}  macro_f1: {}  nmi: {}  ari: {}".format(micro_f1,macro_f1,nmi,ari))
    print("micro_f1: {}  macro_f1: {}  nmi: {}  ari: {}".format(micro_f1,macro_f1,nmi,ari))

    return micro_f1

def parse_adjlist(adjlist, edge_metapath_indices, samples=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                neighbors = row_parsed[1:]
                result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                neighbors = [row_parsed[i + 1] for i in sampled_idx]
                result_indices.append(indices[sampled_idx])
        else:
            neighbors = []
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse(adjlists, edge_metapath_indices_list, idx_batch, device, samples=None):
    g_list = []
    result_indices_list = []
    idx_batch_mapped_list = []
    for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
        edges, result_indices, num_nodes, mapping = parse_adjlist(
            [adjlist[i] for i in idx_batch], [indices[i] for i in idx_batch], samples)
        #TODO
        g = dgl.DGLGraph(multigraph=True).to(device)
        g.add_nodes(num_nodes)
        if len(edges) > 0:
            sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
        else:
            result_indices = torch.LongTensor(result_indices).to(device)
        #g.add_edges(*list(zip(*[(dst, src) for src, dst in sorted(edges)])))
        #result_indices = torch.LongTensor(result_indices).to(device)
        g_list.append(g)
        result_indices_list.append(result_indices)
        idx_batch_mapped_list.append(np.array([mapping[idx] for idx in idx_batch]))

    return g_list, result_indices_list, idx_batch_mapped_list