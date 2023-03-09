import os
import sys
import torch
import torch.nn.functional as F

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]
sys.path.append(root_path + 'src')
os.chdir(root_path)
import argparse
import time 
import random
from Model_hyper import GCN_RE
from dataset_hyper import load_DBLP_data,load_ACM_data,load_Yelp_data
import numpy as np
from loguru import logger
from tools import parse, test

seed = 0 #DBLP 
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

def train(args, gpu_id=0):
    
    device = torch.device("cuda:" + str(gpu_id) if gpu_id >= 0 else "cpu")
    args.device = device
    # Load Graph
    if args.dataset == "DBLP":
        hyperedge, metapath_adjs, features_list, type_mask, labels, pos = load_DBLP_data()

        hyperedge = [torch.LongTensor(hyper).to(device) for hyper in hyperedge]
        features_list = [torch.FloatTensor(features).to(device) for features in features_list]
        metapath_adjs = [adj.to(device) for adj in metapath_adjs]
        type_mask = torch.LongTensor(type_mask).to(device)
        pos = pos.to(device)

        args.edges_num = int(hyperedge[0][1].max() + 1)
        args.in_dims = [features.shape[1] for features in features_list]
        args.num_metapaths = len(metapath_adjs)
        
    elif args.dataset == "ACM":
        hyperedge, metapath_adjs, features_list, type_mask, labels, pos = load_ACM_data()

        hyperedge = [torch.LongTensor(hyper).to(device) for hyper in hyperedge]
        features_list = [torch.FloatTensor(features).to(device) for features in features_list]
        metapath_adjs = [adj.to(device) for adj in metapath_adjs]
        type_mask = torch.LongTensor(type_mask).to(device)
        
        args.edges_num = int(hyperedge[0][1].max() + 1)
        pos = pos.to(device)
        args.in_dims = [features.shape[1] for features in features_list]
        args.num_metapaths = len(metapath_adjs)

    elif args.dataset == "Yelp":
        hyperedge, metapath_adjs, features_list, type_mask, labels, pos = load_Yelp_data()

        hyperedge = [torch.LongTensor(hyper).to(device) for hyper in hyperedge]
        features_list = [torch.FloatTensor(features).to(device) for features in features_list]
        metapath_adjs = [adj.to(device) for adj in metapath_adjs]
        type_mask = torch.LongTensor(type_mask).to(device)
        pos = pos.to(device)

        args.edges_num = int(hyperedge[0][1].max() + 1)
        # args.metapath_dim = mps_embed.shape[1]
        args.in_dims = [features.shape[1] for features in features_list]
        args.num_metapaths = len(metapath_adjs)
        # args.num_metapaths = 1
    
   
    labels = torch.LongTensor(labels).to(device)
    logger.info(args.dataset + ' data load finish')
    
    # ! Train Init
    logger.info(f'{args}\nStart training..')

    model = GCN_RE(args) 
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    inputs = (hyperedge, metapath_adjs, features_list, type_mask)

    dur = []
    cnt_wait = 0
    best_mif1 = 0
    # loss_cnt = 1e9
    if args.test == False:
        for epoch in range(1,args.epochs):
            # ! Train
            t0 = time.time()
            model.train()

            
            loss = model.loss(inputs, pos)  
            
            dur.append(time.time() - t0)
            logger.info(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} | Loss {loss.item():.4f}")

            if epoch % args.eval == 0:
                model.eval()
                embeds = model.get_embeds(inputs).cpu()
                micro_f1 = test(embeds,labels.cpu())
                if best_mif1 < micro_f1:
                    best_mif1 = micro_f1
                    cnt_wait = 0
                    torch.save(model.state_dict(),'./Model/{}/Model_{}.pkl'.format(args.dataset,args.dataset))
                else:
                    cnt_wait += 1
                
            if cnt_wait == args.patience:
                logger.info("Early stopping!")
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
    model.load_state_dict(torch.load('./Model/{}/Model_{}.pkl'.format(args.dataset,args.dataset)))
    model.eval()

    embeds = model.get_embeds(inputs).cpu()
    emb_show = embeds.clone()
    emb_show = emb_show.to("cpu").numpy()
    np.save("{}_features.npy".format(args.dataset),emb_show)

    test(embeds,labels.cpu())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dataset = 'DBLP'
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--eval', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    if dataset == "DBLP":
        parser.add_argument('--lr', type=float, default=0.001) 
        parser.add_argument('--patience', type=int, default=5) # DBLP
        parser.add_argument('--aggre_method', type=str, default="mean")
        parser.add_argument('--target', type=str, default="Authors") 
        parser.add_argument('--nheads', type=int, default=1)
    elif dataset == "ACM":
        parser.add_argument('--patience', type=int, default=5) # ACM
        parser.add_argument('--aggre_method', type=str, default="mean")
        parser.add_argument('--target', type=str, default="Papers") 
        parser.add_argument('--lr', type=float, default=0.001) 
        parser.add_argument('--nheads', type=int, default=2)
    elif dataset == "Yelp":
        parser.add_argument('--patience', type=int, default=5) # YELP
        parser.add_argument('--aggre_method', type=str, default="concat")
        parser.add_argument('--target', type=str, default="Bussiness") 
        parser.add_argument('--lr', type=float, default=0.001) 
        parser.add_argument('--nheads', type=int, default=2)


    parser.add_argument('--epochs', type=int, default=1000) 
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--out-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--alpha', type=float, default=0.2)
    
    args = parser.parse_args()
    
    args.test = False
    
    t = time.strftime('%Y-%m-%d %H:%M', time.localtime())
    logger.add("./log/{}/{}_{}.txt".format(args.dataset, args.dataset, t))
    
    train(args,args.gpu_id)
    

 