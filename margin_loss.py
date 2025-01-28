import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.config as config
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.clustering import get_margins, obtain_and_evaluate_clusters
from utils.dataset import CelebaDataset, WaterBirds
from utils.utils import compute_accuracy, save_state_dict

from models.basemodel import Network, NetworkMargin


def parse_args():
    # Parse the arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='baseline',
                        help='baseline or adversarial')
    parser.add_argument('--dataset', type=str, default='celeba',
                        help='which dataset to train on?')   
    parser.add_argument('--bias', action='store_true',
                        help='bias-amplify the model?')           
    parser.add_argument('--clustering', action='store_true',
                        help='only cluster')
    parser.add_argument('--train', action='store_true',
                        help='train, eval, test')
    parser.add_argument('--val-only', action='store_true',
                        help='evaluate on the val set one time')
    parser.add_argument('--test_only', action='store_true',
                        help='evaluate on the test set one time')
    parser.add_argument("--gpu", type=str, default='0',
                        help='gpu card ID')
    parser.add_argument('--seed', type=int, default=2411,
                        help='seed to run')
    args = parser.parse_args()
    return args


def read_data(args):
    # Create the train, test and val loaders
    
    batch_size = config.base_batch_size
    if args.train:
        if args.dataset == 'celeba':
            train_dataset = CelebaDataset(split=0)
            valid_dataset = CelebaDataset(split=1)
            test_dataset = CelebaDataset(split=2)

        elif args.dataset == 'waterbirds':
            train_dataset = WaterBirds(split='train')
            valid_dataset = WaterBirds(split='val')
            test_dataset = WaterBirds(split='test')

        if args.bias:
            class_sample_count = train_dataset.class_sample_count
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in train_dataset.targets])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            print(samples_weight.shape)
            shuffle = False
            sampler = sampler
        else:
            shuffle = True
            sampler = None
            
        train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            sampler=sampler,
                            num_workers=4)

        valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

        test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)
        
        return train_loader, valid_loader, test_loader

    elif args.val_only:
        if args.dataset == 'celeba':
            valid_dataset = CelebaDataset(split=1)
        else:
            valid_dataset = WaterBirds(split='val')
        valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)
        return valid_loader
    
    else:
        if args.dataset == 'celeba':
            test_dataset = CelebaDataset(split=2)
        else:
            test_dataset = WaterBirds(split='test')
        test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)
        return test_loader

class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')
  
        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss


def cross_entropy_loss_arc(logits, labels, **kwargs):
    """ Modified cross entropy loss to compute the margin loss"""
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels
    return loss.sum(dim=-1).mean()


def train(model, NUM_EPOCHS, optimizer, DEVICE, train_loader, valid_loader, test_loader, args):
    # training loop
    if args.type == 'margin':
        baseline = Network(config.model_name, config.num_class, config.mlp_neurons, config.hid_dim)

        ''' Comment lines 108-110 only if the bias-amplified model is not required.'''
        model_name = config.basemodel_path
        with torch.no_grad():
            baseline.load_state_dict(torch.load(os.path.join('./', model_name), map_location=DEVICE))

        baseline.eval()
        baseline = baseline.to(DEVICE)
        kmeans, _, all_margins = get_margins(train_loader, baseline, DEVICE)
    
    start_time = time.time()
    best_val = 0
    best_worst, best_avg = 999, 999
    
    for epoch in range(NUM_EPOCHS):
        
        model.train()
        for _, (_, features, targets, z1, _) in enumerate(train_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            z1 = z1.to(DEVICE)
            
            if args.type == 'margin':
                one_hot = F.one_hot(targets).to(DEVICE)
                with torch.no_grad():
                    _, _, feats_baseline = baseline(features)
                    
                feats_baseline = feats_baseline.cpu().detach().numpy()
                pseudo_labels = kmeans.predict(feats_baseline)
            
                margins = all_margins[pseudo_labels]            
                
                margins = torch.from_numpy(margins)
                margins = margins.to(DEVICE)
                features = features.to(torch.float32)
                logits, _, _, _, _ = model(features, margins, s=config.scale)
                
                cost = cross_entropy_loss_arc(logits, one_hot.float())
                
                optimizer.zero_grad()
                cost.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()

            elif args.type == 'baseline':
                logits, _, _ = model(features)
            
                cost = nn.CrossEntropyLoss()(logits, targets.long()) 
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
        
        # Evaluate the run
        model.eval()
        
        with torch.set_grad_enabled(False): # save memory during inference
            
            train_acc, train_worst, train_avg = compute_accuracy(model, train_loader, device=DEVICE)
            val_acc, val_worst, val_avg = compute_accuracy(model, valid_loader, device=DEVICE)
            test_acc, test_worst, test_avg = compute_accuracy(model, test_loader, device=DEVICE)
            if args.type == 'baseline' and args.bias:
                overall_acc = train_acc
            else:
                overall_acc = val_acc
            if best_val < overall_acc:
                print('Model saved at epoch', epoch)
                best_val = overall_acc
                if args.type == 'margin':
                    save_state_dict(model.state_dict(), os.path.join('./', config.margin_path))
                elif args.type == 'baseline':
                    save_state_dict(model.state_dict(), os.path.join('./', config.basemodel_path))
            
                best_worst = test_worst
                best_avg = test_avg
            print('Epoch', epoch)
            print('Train worst, avg, global acc', train_worst, train_avg, train_acc)
            print('Val worst, avg, global acc', val_worst, val_avg, val_acc)
            print('Test worst, avg, global acc', test_worst, test_avg, test_acc)
                
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    print("Final val acc", best_val)
    print("Test Worst:", best_worst)
    print("Test Avg:", best_avg)

    return best_val


def eval(model, data_loader, path):
    
    model.load_state_dict(torch.load(os.path.join('./', path), map_location=DEVICE)) 
    model.eval()

    with torch.no_grad():
        test_acc, test_worst, test_avg = compute_accuracy(model, data_loader, DEVICE, adv=adv, club=True)
        print("Global Acc", test_acc)
        print("Worst:", test_worst)
        print("Avg:", test_avg)

if __name__ == '__main__':
    args = parse_args()
    seed = args.seed
    
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    DEVICE = f'cuda:{str(args.gpu)}'
    
    if args.dataset == 'celeba' or args.dataset == 'waterbirds':
        celeba = True
    else:
        celeba = False
    
    if args.train:
        # For training
        train_loader, valid_loader, test_loader = read_data(args)
        if args.type == 'baseline':
            # Baseline training
            model = Network(config.model_name, config.num_class, config.mlp_neurons, config.hid_dim)
            model.to(DEVICE)
            lr = config.base_lr
            weight_decay = config.weight_decay
            if config.opt_b == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay) #, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            epochs = config.base_epochs
            train(model, config.base_epochs, optimizer, DEVICE, train_loader, valid_loader, test_loader, args)
        elif args.type == 'margin':
            # Margin loss
            model = NetworkMargin(config.model_name, config.num_class, DEVICE, config.std, config.mlp_neurons, config.hid_dim)
            model = model.to(DEVICE)
            lr = config.base_lr
            weight_decay = config.weight_decay
            if config.opt_m == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay) #, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            train(model, config.base_epochs, optimizer, DEVICE, train_loader, valid_loader, test_loader, args)
        
    elif args.clustering:
        # Calculate cluster NMIs
        args.train = True
        train_loader, valid_loader, test_loader = read_data(args)

        baseline = Network(config.model_name, config.num_class, config.mlp_neurons)
        model_name = config.basemodel_path
        with torch.no_grad():
            
            ''' Comment only if you do not want to load '''
            baseline.load_state_dict(torch.load(os.path.join('./', model_name), map_location=DEVICE))
            baseline.to(DEVICE)
            baseline.eval()
            obtain_and_evaluate_clusters(train_loader, baseline, DEVICE)

    elif args.val_only:
        valid_loader = read_data(args)
        
        if args.type == 'baseline':
            model = Network(config.model_name, config.num_class, config.mlp_neurons)
        else:
            model = NetworkMargin(config.model_name, config.num_class, DEVICE, config.mlp_neurons)
        
        model = model.to(DEVICE)
        
        if args.type == 'baseline':
            eval(model, valid_loader, config.baseline_path)
        else:
            eval(model, valid_loader, config.margin_path)

    elif args.test_only:
        test_loader = read_data(args)

        if args.type == 'baseline':
            model = Network(config.model_name, config.num_class, config.mlp_neurons)
        else:
            model = NetworkMargin(config.model_name, config.num_class, DEVICE, config.mlp_neurons)
        
        model = model.to(DEVICE)
        
        if args.type == 'baseline':
            eval(model, test_loader, config.baseline_path)
        else:
            eval(model, test_loader, config.margin_path)
    
    print("VRAM taken: ", torch.cuda.max_memory_allocated() / 1024**2)