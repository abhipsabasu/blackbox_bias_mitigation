import os
import pdb
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import utils.config as config
from PIL import Image
# from models import model_attributes
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import MNIST
from tqdm import tqdm
from utils.utils import save_state_dict

from models.basemodel import Network, NetworkMargin


class BiasedMNIST(MNIST):

    def __init__(self, split):
        # if split == 'val':
        #     split = 'val_testlike'
        root = '/data2/abhipsa/blackbox_bias_mitigation/blackbox-codebase/colormnist_embeddings/res18/0.995'
        
        self.data = np.load(os.path.join(root, split+"_feats.npy")).astype(np.float32)
        self.targets = torch.LongTensor(np.load(os.path.join(root, split+"_targets.npy")))
        self.biased_targets = torch.LongTensor(np.load(os.path.join(root, split+"_bias.npy")))
        self.class_sample_count = np.array(
            [len(np.where(self.targets == t)[0]) for t in np.unique(self.targets)])
        self.cluster_ids = None
        self.clustering = False
        self.sample_weights = None

        self.record = {}
        ctr = 0
        for t in range(10):
            for b in range(10):
                self.record[str(t)+'_'+str(b)] = ctr
                ctr += 1

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def __getitem__(self, index):
        img, target, bias = self.data[index], int(self.targets[index]), int(self.biased_targets[index])
        if self.cluster_ids is not None:
            cluster = self.cluster_ids[index]
        else:
            cluster = -1
            
        if self.sample_weights is not None:
            weight = self.sample_weights[index]
        else:
            weight = 1

        group = self.record[str(target) + '_' + str(bias)]

        if self.clustering is True:
            return img, target, index
        
        return img, target, np.array([bias]), group, cluster, weight, index

    def clustering_on(self):
        self.clustering = True
    
    def clustering_off(self):
        self.clustering = False
    
    def update_clusters(self, cluster_ids):
        self.cluster_ids = cluster_ids   
        
    def update_weights(self, weights):
        self.sample_weights = weights         
    
    def __len__(self):
        return len(self.data)

    @property
    def classes(self):
        return [str(i) for i in range(10)]
    
    @property
    def num_classes(self):
        return len(self.classes)
    
    @property
    def num_groups(self):
        return np.power(len(self.classes), 1+1)
    
    @property
    def bias_attributes(self):
        return

train_dataset = BiasedMNIST(split='train')
val_dataset = BiasedMNIST(split='val')
test_dataset = BiasedMNIST(split='test')
print(len(train_dataset), len(val_dataset), len(test_dataset))
batch_size = config.base_batch_size

loaders = {'train': DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4),
            'valid': DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4),
            'test': DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)}

lr = config.base_lr
weight_decay = config.weight_decay

seed = 3000
gpu = 2
np.random.seed(seed)
random.seed(seed)
print(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic=True
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = f'cuda:{str(gpu)}'

model = Network(config.model_name, config.num_class, config.mlp_neurons)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def compute_supcon_loss(feats, targets, device):
    tau = 1.0
    feats_filt = F.normalize(feats, dim=1)
    targets_r = targets.reshape(-1, 1)
    targets_c = targets.reshape(1, -1)
    mask = targets_r == targets_c
    mask = mask.int().to(device)
    feats_sim = torch.exp(torch.matmul(feats_filt, feats_filt.T) / tau)
    negatives = feats_sim*(1.0 - mask)
    negative_sum = torch.sum(negatives)
    positives = torch.log(feats_sim/negative_sum)*mask
    positive_sum = torch.sum(positives)
    positive_sum = positive_sum/torch.sum(mask)

    sup_con_loss = -1*torch.mean(positive_sum)
    return sup_con_loss

def get_acc(a):
    # a = '100.0%/100.0%/100.0%/90.0%/100.0%/80.0%/90.0%/88.9%/100.0%/100.0%/100.0%/40.0%/0.0%/10.0%/90.0%/80.0%/60.0%/55.6%/30.0%/100.0%/100.0%/20.0%/33.3%/0.0%/60.0%/0.0%/0.0%/44.4%/0.0%/90.0%/90.0%/20.0%/0.0%/10.0%/70.0%/40.0%/0.0%/33.3%/40.0%/90.0%/80.0%/90.0%/55.6%/100.0%/80.0%/100.0%/90.0%/66.7%/80.0%/90.0%/100.0%/40.0%/44.4%/80.0%/100.0%/0.0%/100.0%/0.0%/0.0%/50.0%/90.0%/0.0%/0.0%/0.0%/40.0%/90.0%/0.0%/0.0%/0.0%/0.0%/70.0%/0.0%/0.0%/0.0%/40.0%/0.0%/0.0%/0.0%/0.0%/70.0%/100.0%/20.0%/0.0%/20.0%/80.0%/50.0%/10.0%/0.0%/0.0%/70.0%/100.0%/100.0%/88.9%/90.0%/100.0%/90.0%/40.0%/0.0%/20.0%/100.0%/90.9%/100.0%/90.9%/100.0%/100.0%/90.9%/81.8%/91.7%/100.0%/58.3%/100.0%/100.0%/100.0%/100.0%/100.0%/100.0%/90.9%/100.0%/100.0%/100.0%/27.3%/83.3%/63.6%/9.1%/58.3%/90.9%/100.0%/100.0%/54.5%/58.3%/81.8%/90.9%/100.0%/100.0%/100.0%/100.0%/100.0%/100.0%/100.0%/90.9%/0.0%/33.3%/9.1%/36.4%/16.7%/63.6%/90.9%/50.0%/45.5%/16.7%/90.9%/100.0%/91.7%/100.0%/90.9%/83.3%/90.9%/54.5%/91.7%/27.3%/0.0%/0.0%/45.5%/0.0%/25.0%/9.1%/18.2%/66.7%/63.6%/50.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/54.5%/0.0%/0.0%/0.0%/100.0%/100.0%/100.0%/90.9%/91.7%/90.9%/100.0%/100.0%/72.7%/91.7%/27.3%/63.6%/100.0%/81.8%/100.0%/16.7%/27.3%/9.1%/75.0%/9.1%/0.0%/0.0%/60.0%/10.0%/0.0%/20.0%/20.0%/18.2%/20.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/81.8%/80.0%/100.0%/100.0%/40.0%/90.0%/72.7%/100.0%/100.0%/90.9%/20.0%/0.0%/18.2%/0.0%/0.0%/0.0%/0.0%/70.0%/30.0%/54.5%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/45.5%/0.0%/70.0%/0.0%/0.0%/0.0%/0.0%/9.1%/0.0%/40.0%/0.0%/0.0%/0.0%/63.6%/0.0%/0.0%/0.0%/60.0%/60.0%/0.0%/70.0%/0.0%/9.1%/0.0%/0.0%/0.0%/0.0%/20.0%/0.0%/18.2%/20.0%/20.0%/81.8%/100.0%/0.0%/27.3%/80.0%/40.0%/100.0%/10.0%/10.0%/50.0%/40.0%/90.0%/27.3%/80.0%/50.0%/20.0%/100.0%/50.0%/80.0%/80.0%/100.0%/100.0%/90.9%/20.0%/100.0%/100.0%/80.0%/100.0%/10.0%/70.0%/20.0%/50.0%/72.7%/80.0%/90.0%/0.0%/50.0%/0.0%/90.0%/90.0%/100.0%/100.0%/100.0%/70.0%/100.0%/90.0%/100.0%/90.0%/0.0%/30.0%/90.0%/40.0%/27.3%/20.0%/80.0%/50.0%/10.0%/10.0%/0.0%/0.0%/20.0%/0.0%/63.6%/0.0%/70.0%/0.0%/0.0%/0.0%/100.0%/100.0%/100.0%/100.0%/63.6%/30.0%/70.0%/70.0%/90.0%/50.0%/0.0%/30.0%/50.0%/60.0%/27.3%/20.0%/30.0%/0.0%/10.0%/40.0%/90.0%/100.0%/80.0%/90.0%/45.5%/30.0%/100.0%/80.0%/100.0%/90.0%/30.0%/60.0%/10.0%/40.0%/18.2%/20.0%/90.0%/0.0%/70.0%/0.0%/30.0%/60.0%/0.0%/30.0%/0.0%/20.0%/10.0%/0.0%/11.1%/40.0%/70.0%/0.0%/20.0%/0.0%/60.0%/10.0%/40.0%/50.0%/10.0%/77.8%/50.0%/10.0%/0.0%/0.0%/80.0%/0.0%/0.0%/0.0%/0.0%/0.0%/100.0%/20.0%/0.0%/0.0%/100.0%/0.0%/11.1%/0.0%/0.0%/40.0%/100.0%/100.0%/100.0%/100.0%/100.0%/100.0%/60.0%/100.0%/100.0%/90.0%/40.0%/10.0%/66.7%/10.0%/100.0%/0.0%/90.0%/0.0%/0.0%/0.0%/80.0%/40.0%/50.0%/22.2%/90.0%/90.0%/10.0%/10.0%/20.0%/100.0%/0.0%/0.0%/0.0%/0.0%/10.0%/0.0%/0.0%/0.0%/0.0%/0.0%/22.2%/0.0%/0.0%/10.0%/40.0%/0.0%/44.4%/0.0%/0.0%/10.0%/40.0%/44.4%/0.0%/10.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/11.1%/11.1%/22.2%/11.1%/77.8%/77.8%/62.5%/44.4%/0.0%/22.2%/33.3%/33.3%/0.0%/0.0%/77.8%/100.0%/11.1%/22.2%/62.5%/22.2%/11.1%/0.0%/0.0%/0.0%/11.1%/11.1%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/11.1%/77.8%/0.0%/11.1%/0.0%/0.0%/66.7%/55.6%/0.0%/50.0%/33.3%/88.9%/44.4%/55.6%/66.7%/100.0%/100.0%/88.9%/100.0%/88.9%/100.0%/100.0%/100.0%/100.0%/100.0%/100.0%/55.6%/0.0%/0.0%/0.0%/44.4%/100.0%/0.0%/0.0%/0.0%/55.6%/33.3%/77.8%/22.2%/11.1%/88.9%/100.0%/22.2%/22.2%/0.0%/22.2%/22.2%/25.0%/0.0%/55.6%/77.8%/77.8%/0.0%/0.0%/0.0%/22.2%/44.4%/33.3%/22.2%/75.0%/100.0%/77.8%/100.0%/55.6%/0.0%/0.0%/0.0%/100.0%/40.0%/88.9%/0.0%/88.9%/40.0%/10.0%/55.6%/0.0%/66.7%/40.0%/80.0%/44.4%/60.0%/44.4%/80.0%/0.0%/70.0%/60.0%/33.3%/40.0%/33.3%/30.0%/90.0%/33.3%/100.0%/44.4%/10.0%/44.4%/60.0%/30.0%/11.1%/0.0%/33.3%/10.0%/55.6%/0.0%/10.0%/33.3%/60.0%/100.0%/100.0%/80.0%/33.3%/80.0%/88.9%/90.0%/44.4%/10.0%/10.0%/22.2%/20.0%/55.6%/0.0%/0.0%/80.0%/0.0%/0.0%/0.0%/100.0%/100.0%/100.0%/100.0%/80.0%/100.0%/100.0%/100.0%/100.0%/100.0%/77.8%/90.0%/77.8%/90.0%/100.0%/88.9%/100.0%/11.1%/90.0%/66.7%/80.0%/80.0%/55.6%/70.0%/33.3%/10.0%/100.0%/0.0%/0.0%/66.7%/20.0%/88.9%/50.0%/80.0%/55.6%/50.0%/66.7%/0.0%/44.4%/10.0%/0.0%/9.1%/0.0%/0.0%/10.0%/27.3%/0.0%/10.0%/9.1%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/10.0%/10.0%/0.0%/0.0%/20.0%/9.1%/40.0%/0.0%/10.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/27.3%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/10.0%/0.0%/40.0%/10.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/10.0%/0.0%/18.2%/0.0%/0.0%/10.0%/0.0%/80.0%/0.0%/0.0%/100.0%/100.0%/100.0%/100.0%/90.0%/100.0%/100.0%/100.0%/100.0%/100.0%/0.0%/20.0%/10.0%/54.5%/0.0%/50.0%/0.0%/72.7%/0.0%/0.0%/0.0%/9.1%/50.0%/20.0%/0.0%/40.0%/40.0%/10.0%/9.1%/0.0%/60.0%/77.8%/100.0%/100.0%/30.0%/55.6%/60.0%/50.0%/80.0%/11.1%/50.0%/80.0%/30.0%/77.8%/30.0%/40.0%/20.0%/44.4%/60.0%/80.0%/30.0%/44.4%/70.0%/30.0%/50.0%/55.6%/70.0%/50.0%/66.7%/30.0%/70.0%/50.0%/0.0%/30.0%/20.0%/60.0%/0.0%/40.0%/60.0%/50.0%/44.4%/70.0%/70.0%/60.0%/44.4%/40.0%/10.0%/60.0%/77.8%/10.0%/20.0%/66.7%/60.0%/90.0%/40.0%/0.0%/50.0%/0.0%/50.0%/11.1%/40.0%/80.0%/100.0%/100.0%/70.0%/80.0%/0.0%/77.8%/80.0%/100.0%/60.0%/66.7%/50.0%/80.0%/20.0%/11.1%/0.0%/0.0%/55.6%/30.0%/100.0%/90.0%/100.0%/100.0%/90.0%/100.0%/100.0%/100.0%/100.0%/100.0%/33.3%/90.0%/70.0%/80.0%/88.9%/80.0%/40.0%/30.0%/77.8%/10.0%/90.0%/90.0%/100.0%/80.0%/80.0%/72.7%/90.0%/90.0%/60.0%/100.0%/0.0%/0.0%/0.0%/0.0%/60.0%/0.0%/0.0%/10.0%/0.0%/10.0%/90.0%/0.0%/20.0%/0.0%/30.0%/20.0%/0.0%/45.5%/0.0%/90.0%/20.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/10.0%/0.0%/10.0%/10.0%/10.0%/0.0%/0.0%/0.0%/0.0%/0.0%/40.0%/0.0%/45.5%/0.0%/0.0%/0.0%/0.0%/30.0%/0.0%/0.0%/0.0%/0.0%/10.0%/40.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/20.0%/80.0%/30.0%/0.0%/50.0%/0.0%/60.0%/0.0%/0.0%/0.0%/90.0%/40.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/0.0%/100.0%/100.0%/100.0%/70.0%/100.0%/100.0%/70.0%/90.0%/100.0%/100.0%'
    a = a.replace('%', '').split('/')
    x = list(map(float, a))
    
    aligned, conflict = [], []
    ctr = 0

    for target in range(10):
        for bias in range(10):
            if (target == bias):
                aligned.append(x[ctr])
            elif (target != bias):
                conflict.append(x[ctr])
            ctr += 1

    print(sum(aligned) / len(aligned), len(aligned))
    print(sum(conflict) / len(conflict), len(conflict))
    print((sum(aligned) / len(aligned) + sum(conflict) / len(conflict)) / 2)

    # aa, ac, ca, cc = [], [], [], []
    # ctr = 0

    # for t in range(10):
    #     for lb in range(10):
    #         for rb in range(10):
    #             if (lb == t) and (rb == t):
    #                 aa.append(x[ctr])
    #             elif (lb == t) and (rb != t):
    #                 ac.append(x[ctr])
    #             elif (lb != t) and (rb == t):
    #                 ca.append(x[ctr])
    #             elif (lb != t) and (rb != t):
    #                 cc.append(x[ctr])
    #             else:
    #                 print(t, lb, rb)
    #             ctr += 1

    # print(sum(aa) / len(aa), len(aa))
    # print(sum(ac) / len(ac), len(ac))
    # print(sum(ca) / len(ca), len(ca))
    # print(sum(cc) / len(cc), len(cc))
    # print((sum(aa) / len(aa) + sum(ac) / len(ac) + sum(ca) / len(ca) + sum(cc) / len(cc)) / 4)

max_val_acc = 0
for epoch in range(config.base_epochs):
    model.train()
    # print(epoch, model.training)
    for batch_idx, (img, target, bias, group, _, _, index) in enumerate(loaders['train']):
        img = img.to(DEVICE)
        img = img.to(torch.float32)
        target = target.to(DEVICE)
        bias = bias.to(DEVICE)
        logits, _, f = model(img)
        cost = torch.nn.CrossEntropyLoss()(logits, target.long()) #+ compute_supcon_loss(f, target, DEVICE)
        optimizer.zero_grad()
        # scaler.scale(cost).backward()
        # scaler.step(optimizer)
        # scaler.update()
        cost.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
    model.eval()

    test_envs = ['train', 'test']
    calculate_acc = False
                
    for desc in test_envs:

        loader = loaders[desc]
        
        total_top1, total_top5, total_num = 0.0, 0.0, 0
        
        num_classes = len(loader.dataset.classes)
        num_groups = loader.dataset.num_groups
        
        bias_counts = torch.zeros(num_groups).to(DEVICE)
        bias_top1s = torch.zeros(num_groups).to(DEVICE)
        if desc == 'train':
            calculate_acc = False
        
        with torch.no_grad():
            
            features, labels = [], []
            corrects = []

            for _, (data, target, biases, group, _, _, ids) in enumerate(loaders[desc]):
                data, target, biases, group = data.to(DEVICE), target.to(DEVICE), biases.to(DEVICE), group.to(DEVICE)
                
                B = target.size(0)
                num_groups = np.power(num_classes, biases.size(1)+1)

                results, _, feature = model(data)
                pred_labels = results.argsort(dim=-1, descending=True)
                features.append(feature)
                labels.append(group)

        
                top1s = (pred_labels[:, :1] == target.unsqueeze(dim=-1)).squeeze().unsqueeze(0)
                group_indices = (group==torch.arange(num_groups).unsqueeze(1).long().to(DEVICE))
                
                bias_counts += group_indices.sum(1)
                bias_top1s += (top1s * group_indices).sum(1)
                
                corrects.append(top1s)
                
                total_num += B
                total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                acc1, acc5 = total_top1 / total_num * 100, total_top5 / total_num * 100
                
                bias_accs = bias_top1s / bias_counts * 100

                avg_acc = np.nanmean(bias_accs.cpu().numpy())
                worst_acc = np.nanmin(bias_accs.cpu().numpy())
                
                acc_desc = '/'.join(['{:.1f}%'.format(acc) for acc in bias_accs])
                
                # test_bar.set_description('Eval Epoch [{}/{}] [{}] Bias: {:.2f}%'.format(epoch, config.base_epochs, desc, avg_acc))
                
        if calculate_acc:
            get_acc(acc_desc)
            print(f"New best test {acc1} at {epoch}")
            print('Eval Epoch [{}/{}] [{}] Unbiased: {:.2f}% [{}]'.format(epoch, config.base_epochs, desc, avg_acc, acc_desc))
            print('Total [{}]: Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(desc, acc1, acc5))
        # log = self.logger.info if desc in ['train', 'train_eval'] else self.logger.warning
        
        # print("               {} / {} / {}".format(desc, config.target_attr, self.args.bias_attrs))

        if (desc == 'train') and (acc1 > max_val_acc):
            
            print(f"New best train {acc1} at {epoch}, old acc {max_val_acc}")
            max_val_acc = acc1
            save_state_dict(model.state_dict(), 'mnist_baseline.pth')
            # print('Eval Epoch [{}/{}] [{}] Unbiased: {:.2f}% [{}]'.format(epoch, config.base_epochs, desc, avg_acc, acc_desc))
            print('Total [{}]: Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(desc, acc1, acc5))
            calculate_acc = True
        
            


