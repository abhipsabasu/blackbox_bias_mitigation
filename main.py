import argparse
import utils.config as config
from utils.dataset import CelebaDataset, ColoredMNIST
import torch
import os
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models.basemodel import Network
from models.adv_model import AdvNetwork
from utils.utils import compute_accuracy, save_state_dict, show_tsne
from utils.clustering import alternate_clustering
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='baseline',
                        help='baseline or adversarial')
    parser.add_argument('--dataset', type=str, default='celeba',
                        help='which dataset to train on?')              
    parser.add_argument('--clustering', action='store_true',
                        help='only cluster')
    parser.add_argument('--train', action='store_true',
                        help='train, eval, test')
    parser.add_argument('--val-only', action='store_true',
                        help='evaluate on the val set one time')
    parser.add_argument('--test_only', action='store_true',
                        help='evaluate on the test set one time')
    parser.add_argument('--tsne-baseline', action='store_true',
                        help='visualize feature space of baseline model')
    parser.add_argument('--tsne-adv', action='store_true',
                        help='visualize feature space of adversarial')
    parser.add_argument('--bias-percentage', default=None,
                        help='use the more biased splits')
    parser.add_argument("--tsne-data", type=str, default='train',
                        help='for what data we want to visualize tsne')
    parser.add_argument("--tsne-attr", type=str, default='bias',
                        help='for what data we want to visualize tsne')
    parser.add_argument("--gpu", type=str, default='0',
                        help='gpu card ID')
    parser.add_argument('--seed', type=int, default=5193,
                        help='seed to run')
    parser.add_argument('--create-mnist', type=int, default=5193,
                        help='seed to run')
    args = parser.parse_args()
    return args

def read_data(args):
    if args.type == 'baseline':
        batch_size = config.base_batch_size
    else:
        batch_size = config.adv_batch_size
    bias_percentage = None
    if args.bias_percentage:
        print('Loading Biased data')
        bias_percentage = args.bias_percentage
    if args.train:
        if args.dataset == 'celeba':
            train_dataset = CelebaDataset(split=0, bias_percentage=bias_percentage)

            valid_dataset = CelebaDataset(split=1, bias_percentage=bias_percentage)

            test_dataset = CelebaDataset(split=2, bias_percentage=bias_percentage)
        else:
            train_dataset = ColoredMNIST(split=0)

            valid_dataset = ColoredMNIST(split=1)

            test_dataset = ColoredMNIST(split=2)
            

        train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
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
            valid_dataset = ColoredMNIST(split=1)
        valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)
        return valid_loader
    else:
        if args.dataset == 'celeba':
            test_dataset = CelebaDataset(split=2)
        else:
            test_dataset = ColoredMNIST(split=2)
        test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)
        return test_loader


def train(model, NUM_EPOCHS, optimizer, DEVICE, train_loader, valid_loader, test_loader, pseudo_labels=None, adv=False, celeba=True):
    start_time = time.time()
    best_val, ba_ba, ba_bc, bc_ba, bc_bc = 0,0,0,0,0
    best_dp, best_deo, best_val_cls_wise = 999, 999, 0
    best_val_epoch, best_dp_epoch, best_deo_epoch, best_val_cls_epoch = 0, 0, 0, 0
    for epoch in range(NUM_EPOCHS):
        
        model.train()
        for batch_idx, (name, features, targets, z1) in enumerate(train_loader):
            if adv:
                p = float(batch_idx + epoch * len(train_loader)) / NUM_EPOCHS / len(train_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            z1 = z1.to(DEVICE)
            ### FORWARD AND BACK PROP
            if not adv:
                logits, probas, _ = model(features)
                cost = F.binary_cross_entropy_with_logits(logits.squeeze(), targets.float())
            else:
                logits, probas, z_logits, _ = model(features, alpha)
                if pseudo_labels is not None: 
                    u = np.zeros((z1.shape[0],))
                    v = np.zeros((z1.shape[0],))
                    for j in range(z1.shape[0]): 
                        try:
                            u[j] = pseudo_labels[name[j].item()][0]
                            v[j] = pseudo_labels[name[j].item()][1]  
                        except:  
                            u[j] = pseudo_labels[name[j]][0]
                            v[j] = pseudo_labels[name[j]][1]
                    u = torch.from_numpy(u).to(DEVICE)
                    v = torch.from_numpy(v).to(DEVICE)
                    if config.take_c1:
                        z1_cost = F.binary_cross_entropy_with_logits(z_logits.squeeze(), u.float())
                    else:
                        z1_cost = F.binary_cross_entropy_with_logits(z_logits.squeeze(), v.float())
                else:
                    z1_cost = F.binary_cross_entropy_with_logits(z_logits.squeeze(), z1[:, config.mnist_bias].float())
                
                cost = F.binary_cross_entropy_with_logits(logits.squeeze(), targets.float()) + config.lambda_adv * z1_cost
            optimizer.zero_grad()
            
            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                    %(epoch+1, NUM_EPOCHS, batch_idx, 
                        len(train_loader), cost))

        model.eval()
        
        with torch.set_grad_enabled(False): # save memory during inference
            if celeba:
                train_acc, train_mAP, train_f1_male, train_f1_female, train_dp, train_deo = compute_accuracy(model, train_loader, device=DEVICE, adv=adv)
                val_acc, val_mAP, val_f1_male, val_f1_female, val_dp, val_deo = compute_accuracy(model, valid_loader, device=DEVICE, adv=adv)
                train_class_wise = (train_f1_male + train_f1_female) / 2
                val_class_wise = (val_f1_male + val_f1_female) / 2
                if best_val < val_acc:
                    if not adv:
                        save_state_dict(model.state_dict(), os.path.join('./', config.basemodel_path))
                    else:
                        save_state_dict(model.state_dict(), os.path.join('./', config.adv_path))
                    best_val = val_acc
                    best_dp = val_dp
                    best_deo = val_deo
                    best_val_cls_wise = val_class_wise
                    best_val_epoch = epoch
                    print('Model saved')
                print('Epoch: %03d/%03d | Train: %.3f%%, %.3f | Valid: %.3f%%, %.3f' % (
                    epoch+1, NUM_EPOCHS, 
                    train_acc, train_mAP,
                    val_acc, val_mAP))
                print('Train dp, deo', train_dp, train_deo)
                print('Val dp, deo', val_dp, val_deo)
                print('Train acc cls wise diff', train_class_wise)
                print('Val acc cls wise diff', val_class_wise)
            else:
                train_acc, _, _, _, _ = compute_accuracy(model, train_loader, DEVICE, adv=adv, celeba=False)
                val_acc, ba_ba_new, ba_bc_new, bc_ba_new, bc_bc_new = compute_accuracy(model, valid_loader, DEVICE, adv=adv, celeba=False)
                if best_val < val_acc:
                    if not adv:
                        save_state_dict(model.state_dict(), os.path.join('./', config.basemodel_path))
                    else:
                        save_state_dict(model.state_dict(), os.path.join('./', config.adv_path))
                    best_val = val_acc
                    ba_ba = ba_ba_new
                    ba_bc = ba_bc_new
                    bc_ba = bc_ba_new
                    bc_bc = bc_bc_new
                    best_val_epoch = epoch
                print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
                    epoch+1, NUM_EPOCHS, 
                    train_acc,
                    val_acc))
                print(ba_ba_new, ba_bc_new, bc_ba_new, bc_bc_new)
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    print("Val:", best_val, best_val_epoch)
    if celeba:
        print("Val class wise avg:", best_val_cls_wise)
        print("DP:", best_dp)
        print("DEO:", best_deo)
    else:
        print(ba_ba, ba_bc, bc_ba, bc_bc)
        
    eval(model, test_loader, adv, celeba=celeba)


def eval(model, data_loader, adv, celeba=True):
    if not adv:
        model.load_state_dict(torch.load(os.path.join('./', config.basemodel_path)))
    else:
        model.load_state_dict(torch.load(os.path.join('./', config.adv_path))) 
    model.eval()
    if celeba:
        test_acc, test_mAP, test_f1_male, test_f1_female, test_dp, test_deo = compute_accuracy(model, data_loader, DEVICE, adv=adv)
        best_test_cls_wise = (test_f1_male + test_f1_female) / 2
        print("Test class wise avg:", best_test_cls_wise)
        print("DP:", test_dp)
        print("DEO:", test_deo)
    else:
        test_acc, ba_ba_new, ba_bc_new, bc_ba_new, bc_bc_new = compute_accuracy(model, data_loader, DEVICE, adv=adv, celeba=False)
        _, _, _, _, test_dp0, test_deo0 = compute_accuracy(model, data_loader, DEVICE, adv=adv, 
                                                                                bias_idx=0)
        _, _, _, _, test_dp1, test_deo1 = compute_accuracy(model, data_loader, DEVICE, adv=adv, 
                                                                                bias_idx=1)
        print(ba_ba_new, ba_bc_new, bc_ba_new, bc_bc_new)
        print("Left color - dp:", test_dp0, "deo:", test_deo0)
        print("Right color - dp:", test_dp1, "deo:", test_deo1)
    print("Test:", test_acc)
    


if __name__ == '__main__':
    args = parse_args()
    seed = args.seed
    np.random.seed(seed)
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    DEVICE = 'cuda:{}'.format(args.gpu)
    if args.dataset == 'celeba':
        celeba = True
    else:
        celeba = False
    cudnn.benchmark = True
    if args.train:
        train_loader, valid_loader, test_loader = read_data(args)
        if args.type == 'baseline':
            model = Network(config.model_name, config.num_class)
            model.to(DEVICE)
            lr = config.base_lr
            weight_decay = config.weight_decay
            optimizer = torch.optim.Adam(list(model.classifier.parameters()) + list(model.new_feats.parameters()), 
                             lr=lr, weight_decay = weight_decay)
            epochs = config.base_epochs
            train(model, config.base_epochs, optimizer, DEVICE, train_loader, valid_loader, test_loader, celeba=celeba)
        else:
            model = AdvNetwork(config.model_name, config.num_class)
            model.to(DEVICE)
            baseline = Network(config.model_name, config.num_class)
            baseline = baseline.to(DEVICE)
            lr = config.adv_lr
            weight_decay = config.adv_weight_decay
            optimizer = torch.optim.Adam(list(model.classifier.parameters())+list(model.z_classifier.parameters()) +
                             list(model.new_feats.parameters()), 
                             lr=lr, weight_decay = weight_decay)
            if args.clustering:
                pseudo_labels = alternate_clustering(train_loader, baseline, DEVICE)
            else:
                pseudo_labels = None
            train(model, config.base_epochs, optimizer, DEVICE, train_loader, valid_loader, test_loader, pseudo_labels, adv=True,
                        celeba=celeba)

    elif args.clustering:
        args.train = True
        train_loader, valid_loader, test_loader = read_data(args)
        # model = AdvNetwork(config.model_name, config.num_class)
        # model.to(DEVICE)
        baseline = Network(config.model_name, config.num_class)
        baseline = baseline.to(DEVICE)
        pseudo_labels = alternate_clustering(train_loader, baseline, DEVICE)

    elif args.val_only:
        valid_loader = read_data(args)
        if args.type == 'baseline':
            model = Network(config.model_name, config.num_class)
        else:
            model = AdvNetwork(config.model_name, config.num_class)
        model = model.to(DEVICE)
        if args.type == 'baseline':
            eval(model, valid_loader, False, celeba=celeba)
        else:
            eval(model, valid_loader, True, celeba=celeba)

    elif args.test_only:
        test_loader = read_data(args)
        if args.type == 'baseline':
            model = Network(config.model_name, config.num_class)
        else:
            model = AdvNetwork(config.model_name, config.num_class)
        model = model.to(DEVICE)
        if args.type == 'baseline':
            eval(model, test_loader, False, celeba=celeba)
        else:
            eval(model, test_loader, True, celeba=celeba)
    if args.tsne_adv:
        args.train = True
        train_loader, valid_loader, test_loader = read_data(args)
        model = AdvNetwork(config.model_name, config.num_class)
        model.load_state_dict(torch.load(os.path.join('./', config.adv_path)))
        model.eval()
        model = model.to(DEVICE)
        if args.tsne_data == 'train':
            show_tsne(model, train_loader, DEVICE, args.tsne_attr, True, args.tsne_data)
        elif args.tsne_data == 'test':
            show_tsne(model, test_loader, DEVICE, args.tsne_attr, True, args.tsne_data)
        else:
            show_tsne(model, valid_loader, DEVICE, args.tsne_attr, True, args.tsne_data)
    
    elif args.tsne_baseline:
        args.train = True
        train_loader, valid_loader, test_loader = read_data(args)
        model = Network(config.model_name, config.num_class)
        model.load_state_dict(torch.load(os.path.join('./', config.basemodel_path)))
        model.eval()
        model = model.to(DEVICE)
        if args.tsne_data == 'train':
            show_tsne(model, train_loader, DEVICE, args.tsne_attr, False, args.tsne_data)
        elif args.tsne_data == 'test':
            show_tsne(model, test_loader, DEVICE, args.tsne_attr, False, args.tsne_data)
        else:
            show_tsne(model, valid_loader, DEVICE, args.tsne_attr, False, args.tsne_data)
        



    
