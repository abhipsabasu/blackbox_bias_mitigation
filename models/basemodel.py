import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    # Baseline Network
    def __init__(self, model_name, num_classes, mlp_neurons=None):
        super(Network, self).__init__()
        self.num_classes = num_classes
        self.layer_1 = nn.Sequential(
                        nn.Linear(512, 128),
                        nn.ReLU()
                    )
        self.classifier = nn.Linear(128, self.num_classes+1)

    def forward(self, feats_x):
        # feats_x are the previously stored blackbox features
        x1 = self.layer_1(feats_x)
        logits = self.classifier(x1)
        probas = nn.Softmax(dim=1)(logits)
        return logits, probas[:, 1], F.normalize(x1)



class NetworkMargin(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, model_name, num_classes, DEVICE, mlp_neurons=None, easy_margin=None):
        super(NetworkMargin, self).__init__()
        self.num_classes = num_classes

        self.new_feats = nn.Sequential(
                        nn.Linear(512, mlp_neurons),
                        nn.ReLU(),
                )

        self.s = 1
        self.weight1 = nn.Parameter(torch.FloatTensor(num_classes+1, mlp_neurons))
        nn.init.xavier_uniform_(self.weight1)
        self.device = DEVICE
        self.std = 0.2 # Set the Gaussian randomization standard deviation here
        self.easy_margin = easy_margin

    def forward(self, feats_x, m=None, s=None):
        if s is not None:
            self.s = s

        # feats_x are the previously stored blackbox features
        x = self.new_feats(feats_x)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight1))
        probas = F.softmax(cosine, 1)
        probas = probas[:, 1]
        
        if self.training is False:
            return cosine, probas, F.normalize(x)
        
        if isinstance(m, float):
            m = torch.ones(cosine.shape) * m

        m = m.to(self.device)
        m = torch.normal(mean=m, std=self.std)
        m = 1 - m

        # Compute the angular margins and the corresponding logits for CAML
        self.cos_m = torch.cos(m)
        self.sin_m = torch.sin(m)
        self.th = torch.cos(math.pi - m)
        self.th = self.th.to(self.device)
        self.mm = torch.sin(math.pi - m) * m
        self.mm = self.mm.to(self.device)

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = cosine.to(self.device)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        sine = sine.to(self.device)
        self.cos_m = self.cos_m.to(self.device)
        self.sin_m = self.sin_m.to(self.device)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = phi*self.s
        
        return output, probas, x, cosine, None