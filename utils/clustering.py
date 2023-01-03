from sklearn.cluster import KMeans
import numpy as np
import torch
import utils.config as config
import os


def alternate_clustering(train_loader, model_old, DEVICE):
    overall_feats, overall_targets, overall_z1, mean = extract_clusterFeatures(train_loader, model_old, DEVICE)
    lamb = config.lambda_cluster
    # overall_feats = overall_feats_red
    kmeans = KMeans(n_clusters=2).fit(overall_feats)
    M = kmeans.cluster_centers_
    final_mu = None
    final_delta = None
    cluster_labels_a = (overall_targets == 1).astype(int).squeeze()
    cluster_labels_b = np.random.randint(2, size=cluster_labels_a.shape[0])
    old = overall_feats.shape[0] // 2
    gender_0 = overall_targets[cluster_labels_a == 0]
    males_0 = (gender_0 == 1).sum()
    females_0 = (gender_0 == 0).sum()
    gender_1 = overall_targets[cluster_labels_a == 1]
    males_1 = (gender_1 == 1).sum()
    females_1 = (gender_1 == 0).sum()
    print('Inside cluster 0:', males_0/(males_0 + females_0), females_0/(males_0 + females_0), (males_0 + females_0))
    print('Inside cluster 1:', males_1/(males_1 + females_1), females_1/(males_1 + females_1), (males_1 + females_1))
    for i in range(100):
        print(i)
        M = kmeans.cluster_centers_
        c1_b = overall_feats[cluster_labels_b == 0]
        c2_b = overall_feats[cluster_labels_b == 1]
        c1_a = overall_feats[cluster_labels_a == 0]
        c2_a = overall_feats[cluster_labels_a == 1]
        new = c1_b.shape[0]
        if abs(new - old) <= 10:
            break
        old = new
        print(c1_a.shape, c2_a.shape, c1_b.shape, c2_b.shape)
        V = np.zeros_like(M)
        V[0] = c1_b.mean(axis=0)
        V[1] = c2_b.mean(axis=0)
        M[0] = c1_a.mean(axis=0)
        M[1] = c2_a.mean(axis=0)
        V = V.T
        M = M.T
        Sigma, Q = np.linalg.eig(np.dot(V.T, V))
        Sigma = np.diag(Sigma)
        eta = np.array([lamb/c1_a.shape[0], lamb/c2_a.shape[0]])
        mu = np.zeros_like(M.T)
        Lambda, U = np.linalg.eig(np.dot(M.T, M))
        Lambda = np.diag(Lambda)
        zeta = np.array([lamb/c1_b.shape[0], lamb/c2_b.shape[0]])
        delta = np.zeros_like(M.T)
        print(eta, zeta)
        mu[0] = np.dot(np.identity(M.shape[0]) - eta[0] * np.dot(np.dot(np.dot(V, Q), np.linalg.inv(np.identity(M.shape[1]) + eta[0] * Sigma)), np.dot(Q.T, V.T)), 
                    np.expand_dims(M[:, 0], 1)).squeeze()
        mu[1] = np.dot(np.identity(M.shape[0]) - eta[1] * np.dot(np.dot(np.dot(V, Q), np.linalg.inv(np.identity(M.shape[1]) + eta[1] * Sigma)), np.dot(Q.T, V.T)), 
                    np.expand_dims(M[:, 1], 1)).squeeze()
        delta[0] = np.dot(np.identity(M.shape[0]) - zeta[0] * np.dot(np.dot(np.dot(M, U), np.linalg.inv(np.identity(M.shape[1]) + zeta[0] * Lambda)), np.dot(U.T, M.T)), 
                        np.expand_dims(V[:, 0], 1)).squeeze()
        delta[1] = np.dot(np.identity(M.shape[0]) - zeta[1] * np.dot(np.dot(np.dot(M, U), np.linalg.inv(np.identity(M.shape[1]) + zeta[1] * Lambda)), np.dot(U.T, M.T)), 
                        np.expand_dims(V[:, 1], 1)).squeeze()
        for j in range(cluster_labels_a.shape[0]):
            d1_a = np.linalg.norm(overall_feats[j] - mu[0])
            d2_a = np.linalg.norm(overall_feats[j] - mu[1])
            if d1_a > d2_a:
                cluster_labels_a[j] = 1
            else:
                cluster_labels_a[j] = 0
            d1_b = np.linalg.norm(overall_feats[j] - delta[0])
            d2_b = np.linalg.norm(overall_feats[j] - delta[1])
            if d1_b > d2_b:
                cluster_labels_b[j] = 1
            else:
                cluster_labels_b[j] = 0
        c1_b = overall_feats[cluster_labels_b == 0]
        c2_b = overall_feats[cluster_labels_b == 1]
        c1_a = overall_feats[cluster_labels_a == 0]
        c2_a = overall_feats[cluster_labels_a == 1]
        final_mu = mu
        final_delta = delta
    print('Evaluating on z')
    evaluate_cluster(cluster_labels_a, cluster_labels_b, overall_z1)
    print('Evaluating on y')
    evaluate_cluster(cluster_labels_a, cluster_labels_b, overall_targets)
    return get_pseudolabels(model_old, mean, DEVICE, train_loader, final_mu, final_delta)


def get_pseudolabels(model_old, mean, DEVICE, train_loader, final_mu, final_delta):
    pseudo_labels = dict()
    for batch_idx, (name, features, _, z1) in enumerate(train_loader):
        features = features.to(DEVICE)
        _, _, x = model_old(features)
        feats_numpy = x.cpu().detach().numpy()
        feats_numpy = feats_numpy - mean
        u = np.zeros((feats_numpy.shape[0],))
        for j in range(z1.shape[0]):
            d1_b = np.linalg.norm(feats_numpy[j] - final_mu[0])
            d2_b = np.linalg.norm(feats_numpy[j] - final_mu[1])
            if d1_b > d2_b:
                u[j] = 1
            else:
                u[j] = 0
        v = np.zeros((feats_numpy.shape[0],))
    #         u = gender.cpu().detach().numpy()
        for j in range(z1.shape[0]):
            d1_b = np.linalg.norm(feats_numpy[j] - final_delta[0])
            d2_b = np.linalg.norm(feats_numpy[j] - final_delta[1])
            if d1_b > d2_b:
                v[j] = 1
            else:
                v[j] = 0
        for j in range(z1.shape[0]):
            try:
                pseudo_labels[name[j].item()] = [u[j], v[j]]
            except:
                pseudo_labels[name[j]] = [u[j], v[j]]
    return pseudo_labels


def extract_clusterFeatures(train_loader, model_old, DEVICE):
    overall_feats = None
    overall_z1 = None
    overall_targets = None
    model_old.eval()
    model_name = config.basemodel_path
    model_old.load_state_dict(torch.load(os.path.join('./', model_name)))
    for batch_idx, (_, features, targets, z1) in enumerate(train_loader):
        features = features.to(DEVICE)
       
        _, _, features = model_old(features)
        features = features.cpu().detach().numpy()
        
        z1 = z1.unsqueeze(-1)
        z1 = z1.cpu().detach().numpy()
        z1 = z1[:, config.mnist_bias]
        targets = targets.unsqueeze(-1)
        targets = targets.cpu().detach().numpy()
    
        if overall_feats is None:
            overall_feats = features
            overall_z1 = z1
            overall_targets = targets
    
        else:
            overall_feats = np.vstack((overall_feats, features))
            overall_z1 = np.vstack((overall_z1, z1))
            overall_targets = np.vstack((overall_targets, targets))
    
    mean = overall_feats.mean(axis=0)
    mean = np.expand_dims(mean, axis=0)
    overall_feats = overall_feats - mean
    return overall_feats, overall_targets, overall_z1, mean


def evaluate_cluster(cluster_labels_a, cluster_labels_b, overall_z1):
    gender_0 = overall_z1[cluster_labels_a == 0]
    males_0 = (gender_0 == 1).sum()
    females_0 = (gender_0 == 0).sum()
    gender_1 = overall_z1[cluster_labels_a == 1]
    males_1 = (gender_1 == 1).sum()
    females_1 = (gender_1 == 0).sum()
    print("C1")
    print('Inside cluster 0:', males_0/(males_0 + females_0), females_0/(males_0 + females_0), (males_0 + females_0))
    print('Inside cluster 1:', males_1/(males_1 + females_1), females_1/(males_1 + females_1), (males_1 + females_1))

    gender_0 = overall_z1[cluster_labels_b == 0]
    males_0 = (gender_0 == 1).sum()
    females_0 = (gender_0 == 0).sum()
    gender_1 = overall_z1[cluster_labels_b == 1]
    males_1 = (gender_1 == 1).sum()
    females_1 = (gender_1 == 0).sum()
    print("C2")
    print('Inside cluster 0:', males_0/(males_0 + females_0), females_0/(males_0 + females_0), (males_0 + females_0))
    print('Inside cluster 1:', males_1/(males_1 + females_1), females_1/(males_1 + females_1), (males_1 + females_1))