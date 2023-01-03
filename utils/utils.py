import numpy as np
import torch
import utils.config as config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA


def demographic_parity(y_, u):

    g, uc = np.zeros([2]), np.zeros([2])
    for i in range(u.shape[0]):
        if u[i] > 0:
            g[1] += y_[i]
            uc[1] += 1
        else:
            g[0] += y_[i]
            uc[0] += 1
    g = g / uc
    return np.abs(g[0] - g[1])


def equalized_odds(y, y_, u):
    g = np.zeros([2, 2])
    uc = np.zeros([2, 2])
    for i in range(u.shape[0]):
        if u[i] > 0:
            g[int(y[i])][1] += y_[i]
            uc[int(y[i])][1] += 1
        else:
            g[int(y[i])][0] += y_[i]
            uc[int(y[i])][0] += 1
    acc = g / uc
    acc[0, :] = 1 - acc[0, :]
    worst = np.min(acc)
    avg = np.mean(acc)
    print("Worst group acc:", worst)
    print("Avg group acc:", avg)
    g = g / uc
    return np.abs(g[0, 1] - g[0, 0]) + np.abs(g[1, 1] - g[1, 0]),  worst, avg

def save_state_dict(state_dict, save_path):
    torch.save(state_dict, save_path)

def compute_group(gender, y, labels, y_val, bias_val):
    ba_ba = labels[(gender[:, 0] == bias_val) & (gender[:, 1] == bias_val) & (y == y_val)]
    ba_bc = labels[(gender[:, 0] == bias_val) & (gender[:, 1] == 1-bias_val) & (y == y_val)]
    bc_ba = labels[(gender[:, 0] == 1-bias_val) & (gender[:, 1] == bias_val) & (y == y_val)]
    bc_bc = labels[(gender[:, 0] == 1-bias_val) & (gender[:, 1] == 1-bias_val) & (y == y_val)]

    return ba_ba, ba_bc, bc_ba, bc_bc


def compute_group_numerator(ba_ba_pred, ba_bc_pred, bc_ba_pred, bc_bc_pred, ba_ba_true, ba_bc_true, bc_ba_true, bc_bc_true):
    bias_acc = np.zeros((2, 2))
    bias_acc[0, 0] = (ba_ba_pred == ba_ba_true).sum()
    bias_acc[0, 1] = (ba_bc_pred == ba_bc_true).sum()
    bias_acc[1, 0] = (bc_ba_pred == bc_ba_true).sum()
    bias_acc[1, 1] = (bc_bc_pred == bc_bc_true).sum()
    return bias_acc

def compute_group_denominator(ba_ba_true, ba_bc_true, bc_ba_true, bc_bc_true):
    bias_acc = np.zeros((2, 2))
    bias_acc[0, 0] = ba_ba_true.shape[0]
    bias_acc[0, 1] = ba_bc_true.shape[0]
    bias_acc[1, 0] = bc_ba_true.shape[0]
    bias_acc[1, 1] = bc_bc_true.shape[0]
    return bias_acc


def compute_accuracy(model, data_loader, device, adv=False, celeba=True, bias_idx = None):
    correct_pred, num_examples, correct_male, correct_female, num_males, num_females = 0, 0, 0, 0, 0, 0
    bias_acc = np.zeros((2, 2, 2))
    num_groups = np.zeros((2, 2, 2))
    dp, deo = 0, 0
    pred_total = []
    y_total = []
    gen_total = []
    for i, (_, features, targets, gender) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)
        gender = gender.to(device)
        y = targets.cpu().detach().numpy()
        if not adv:
            logits, probas, _ = model(features)
        else:
            logits, probas, _, _ = model(features, 0)
        predicted_labels = (probas >= 0.5).int().squeeze()
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        class_pred = predicted_labels.cpu().detach().numpy()
        gen = gender.cpu().detach().numpy()
        if bias_idx is not None:
            gen = gen[:, bias_idx]
        if celeba:
            pred_male = class_pred[gen==1]
            pred_female = class_pred[gen==0]
            y_male = y[gen==1]
            y_female = y[gen==0]
            correct_male += (pred_male == y_male).sum()
            correct_female += (pred_female == y_female).sum()
            num_males += y_male.shape[0]
            num_females += y_female.shape[0]
            pred_total = pred_total + class_pred.tolist()
            gen_total = gen_total + gen.tolist()
            y_total = y_total + y.tolist()
        else:
            ba_ba_0_pred, ba_bc_0_pred, bc_ba_0_pred, bc_bc_0_pred = compute_group(gen, y, class_pred, 0, 0)
            ba_ba_0_true, ba_bc_0_true, bc_ba_0_true, bc_bc_0_true = compute_group(gen, y, y, 0, 0)
            ba_ba_1_pred, ba_bc_1_pred, bc_ba_1_pred, bc_bc_1_pred = compute_group(gen, y, class_pred, 1, 1)
            ba_ba_1_true, ba_bc_1_true, bc_ba_1_true, bc_bc_1_true = compute_group(gen, y, y, 1, 1)

            bias_acc[0] += compute_group_numerator(ba_ba_0_pred, ba_bc_0_pred, bc_ba_0_pred, bc_bc_0_pred, 
                                                    ba_ba_0_true, ba_bc_0_true, bc_ba_0_true, bc_bc_0_true)

            bias_acc[1] += compute_group_numerator(ba_ba_1_pred, ba_bc_1_pred, bc_ba_1_pred, bc_bc_1_pred, 
                                                    ba_ba_1_true, ba_bc_1_true, bc_ba_1_true, bc_bc_1_true)

            num_groups[0] += compute_group_denominator(ba_ba_0_true, ba_bc_0_true, bc_ba_0_true, bc_bc_0_true)
            num_groups[1] += compute_group_denominator(ba_ba_1_true, ba_bc_1_true, bc_ba_1_true, bc_bc_1_true)
            
    if celeba:
        pred_total = np.array(pred_total)
        gen_total = np.array(gen_total)
        y_total = np.array(y_total)
        dp = demographic_parity(pred_total, gen_total)
        deo, worst, avg = equalized_odds(y_total, pred_total, gen_total)
        return correct_pred.float()/num_examples * 100, 1, correct_male/num_males, correct_female/num_females, dp, deo
    else:
        bias_acc = bias_acc / num_groups
        avg_accs = (bias_acc[0] + bias_acc[1]) / 2
        return correct_pred.float()/num_examples * 100, avg_accs[0, 0], avg_accs[0, 1], avg_accs[1, 0], avg_accs[1, 1]
    

def show_tsne(model, data_loader, DEVICE, attr_tsne, adv, tsne_data):
    overall_feats = None
    overall_gender = None
    overall_targets = None
    num_correct = 0
    test_length = 0
    for batch_idx, (name, features, targets, gender) in enumerate(data_loader):
        features = features.to(DEVICE)
        if adv:
            _, probas, _, feats = model(features, 0)
        else:
            _, probas, feats = model(features)
        y = targets.cpu().detach().numpy()
        output = probas.detach().cpu().numpy()
        predicted_labels = (output >= 0.5).astype(int).squeeze()
        targets = targets.to(DEVICE)
        gender = gender.to(DEVICE)
        num_correct += (predicted_labels == y).sum()
        feats = feats.cpu().detach().numpy()
        gender = gender.unsqueeze(-1)
        gender = gender.cpu().detach().numpy()
        targets = targets.unsqueeze(-1)
        targets = targets.cpu().detach().numpy()
        if overall_feats is None:
            overall_feats = feats
            overall_gender = gender
            overall_targets = targets
        else:
            overall_feats = np.vstack((overall_feats, feats))
            overall_gender = np.vstack((overall_gender, gender))
            overall_targets = np.vstack((overall_targets, targets))
        test_length += features.shape[0]
    print('Acc', num_correct/test_length)
    print(overall_feats.shape)
    # mean = overall_feats.mean(axis=0)
    # mean = np.expand_dims(mean, axis=0)
    # overall_feats = overall_feats - mean
    overall_gender = overall_gender.squeeze()
    overall_targets = overall_targets.squeeze()

    # overall_feats = overall_feats[:1000]
    # overall_gender = overall_gender[:1000]
    # overall_targets = overall_targets[:1000]

    # tsne = TSNE(2, verbose=1, init='pca')
    # tsne_proj = tsne.fit_transform(overall_feats)
    #tsne_proj = tsne.fit_transform(female_feats)
    pca = PCA(n_components=2)
    tsne_proj = pca.fit_transform(overall_feats)
    print(attr_tsne)
    if attr_tsne == 'bias':
        overall_attr = overall_gender
        label1 = 'attr=1'
        label2 = 'attr=0'
    else:
        overall_attr = overall_targets
        label1 = 'attr=1'
        label2 = 'attr=0'

    indices = (overall_attr == 1)
    x = tsne_proj[indices, 0]
    y = tsne_proj[indices, 1]
    plt.scatter(x, y, label=label1, color='blue')

    indices = (overall_attr == 0)
    x = tsne_proj[indices, 0]
    y = tsne_proj[indices, 1]
    plt.scatter(x, y, label=label2, color='red')
    plt.legend()
    if adv:
        file_name = 'tsne_' + attr_tsne + '_adv_{}.png'.format(tsne_data)
    else:
        file_name = 'tsne_' + attr_tsne + '_baseline_{}.png'.format(tsne_data)
    plt.savefig(os.path.join('Plots', file_name))

