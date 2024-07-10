import numpy as np
import torch


def biased_acc(y, y_, u):
    # Computes worst and avg accuracies
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
    print(acc[0, 0], acc[0, 1], acc[1, 0], acc[1, 1])
    return worst, avg


def save_state_dict(state_dict, save_path):
    # Saves model
    torch.save(state_dict, save_path)


def compute_accuracy(model, data_loader, device):
    # Evaluation for worst and average group accuracies
    correct_pred, num_examples = 0, 0
    pred_total = []
    y_total = []
    gen_total = []
    for _, (_, features, targets, gender, _) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)
        gender = gender.to(device)
        features = features.to(torch.float32)
        y = targets.cpu().detach().numpy()
        
        _, probas, _ = model(features)
        
        predicted_labels = (probas >= 0.5).int().squeeze()
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        class_pred = predicted_labels.cpu().detach().numpy()
        gen = gender.cpu().detach().numpy()
        
        pred_total = pred_total + class_pred.tolist()
        gen_total = gen_total + gen.tolist()
        y_total = y_total + y.tolist()
        
    worst, avg = biased_acc(np.array(y_total), np.array(pred_total), np.array(gen_total))

    return correct_pred.float()/num_examples * 100, worst, avg