import os

import numpy as np
import pandas as pd
import utils.config as config
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from umap import UMAP


def obtain_and_evaluate_clusters(train_loader, model_old, DEVICE):
    # obtain cluster NMIs to identify how well the clusters identify the bias vs the target attributes
    overall_feats, overall_targets, overall_z1, overall_preds, mean = extract_clusterFeatures(train_loader, model_old, DEVICE)
    scores = []
    
    kmeans = KMeans(n_clusters=2, n_init=10).fit(overall_feats)
    kmeans_labels = np.expand_dims(kmeans.labels_, axis=1)

    evaluate_cluster(overall_z1, None, kmeans_labels)
    print()
    # # Calculate for each target label, what is the proportion of the pseudo labels
    evaluate_cluster(overall_targets, None, kmeans_labels)
    
    #Calculate NMIs
    target_nmi = nmi(overall_targets.squeeze().tolist(), kmeans.labels_.tolist())
    bias_nmi = nmi(overall_z1.squeeze().tolist(), kmeans.labels_.tolist())
    print(target_nmi, bias_nmi)
    

def get_margins(train_loader, model_old, DEVICE, kmeans=None):
    # Calculate the margins here. Set K value.
    K = config.K

    overall_feats, overall_targets, overall_z1, overall_preds, _ = extract_clusterFeatures(train_loader, model_old, DEVICE)
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10).fit(overall_feats)
    groups = kmeans.labels_

    target_nmi = nmi(overall_targets.squeeze().tolist(), kmeans.labels_.tolist())
    bias_nmi = nmi(overall_z1.squeeze().tolist(), kmeans.labels_.tolist())
    print(target_nmi, bias_nmi)
    margins = np.zeros((K, 2))
    
    overall_targets = overall_targets.squeeze()
    final_targets = overall_targets
    # overall_preds = overall_preds.squeeze()

    for i in range(K):
        for j in range(2):
            margins[i, j] = final_targets[(groups == i) & (overall_targets == j)].shape[0]
            
    normalized_margins = margins / margins.sum(1, keepdims=True)
    
    print(normalized_margins)
    return kmeans, margins, normalized_margins


def extract_clusterFeatures(train_loader, model_old, DEVICE):
    # Extract features for clustering
    overall_z1 = None
    overall_preds = None
    overall_feats = None
    overall_targets = None
    
    model_old.eval()
    
    for _, (_, img, targets, z1, _) in enumerate(train_loader):
        img = img.to(DEVICE)
        targets = targets.to(DEVICE)
        _, probas, features = model_old(img)
        
        #############################################################
        probas = probas.squeeze()
        predicted_labels = (probas >= 0.5).int().unsqueeze(-1)
        predicted_labels = predicted_labels.cpu().detach().numpy()
        probas = probas.unsqueeze(-1).cpu().detach().numpy()
        #############################################################

        features = features.cpu().detach().numpy()
    
        z1 = z1.unsqueeze(-1)
        z1 = z1.cpu().detach().numpy()

        if z1.shape[1] == 2:
            z1 = z1[:, config.mnist_bias]
        
        targets = targets.unsqueeze(-1)
        targets = targets.cpu().detach().numpy()
    
        if overall_feats is None:
            overall_feats = features
            overall_z1 = z1
            overall_targets = targets
            overall_preds = predicted_labels
    
        else:
            overall_feats = np.vstack((overall_feats, features))
            overall_z1 = np.vstack((overall_z1, z1))
            overall_targets = np.vstack((overall_targets, targets))
            overall_preds = np.vstack((overall_preds, predicted_labels))

    mean = overall_feats.mean(axis=0)
    mean = np.expand_dims(mean, axis=0)
    # overall_feats = overall_feats - mean
    return overall_feats, overall_targets, overall_z1, overall_preds, mean


def evaluate_cluster(cluster_labels_a, cluster_labels_b, overall_z1):
    # Evaluates clusters
    unique_a = np.unique(cluster_labels_a)
    for i in range(unique_a.shape[0]):
        gender_0 = overall_z1[cluster_labels_a == unique_a[i]]
        males_0 = (gender_0 == 1).sum()
        females_0 = (gender_0 == 0).sum()
        print('Inside cluster {}:'.format(unique_a[i]), females_0/(males_0 + females_0), males_0/(males_0 + females_0), (males_0 + females_0))
    
    if cluster_labels_b is not None:
        gender_0 = overall_z1[cluster_labels_b == 0]
        males_0 = (gender_0 == 1).sum()
        females_0 = (gender_0 == 0).sum()
        gender_1 = overall_z1[cluster_labels_b == 1]
        males_1 = (gender_1 == 1).sum()
        females_1 = (gender_1 == 0).sum()
        print("C2")
        print('Inside cluster 0:', females_0/(males_0 + females_0), males_0/(males_0 + females_0), (males_0 + females_0))
        print('Inside cluster 1:', females_1, males_1, (males_1 + females_1))