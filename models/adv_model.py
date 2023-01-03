import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Function
import utils.config as config


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

    
class AdvNetwork(nn.Module):
    def __init__(self, model_name, num_classes):
        super(AdvNetwork, self).__init__()
        self.num_classes = num_classes

        # create cnn model
        model = getattr(models, model_name)(pretrained=True)
        
        #remove fc layers and add a new fc layer
        num_features = model.fc.in_features
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.feats = torch.nn.Sequential(*list(model.children())[:-1])
        self.new_feats = nn.Sequential(
                nn.Linear(num_features, config.mlp_neurons),
                nn.ReLU()   
        )
        self.classifier = nn.Linear(config.mlp_neurons, num_classes)
        self.z_classifier = nn.Linear(config.mlp_neurons, num_classes)
        

    def forward(self, x, alpha):
        x = self.feats(x).squeeze()
        x_new = self.new_feats(x)
        reverse_feature = ReverseLayerF.apply(x_new, alpha)
        logits = self.classifier(x_new)
        z_log = self.z_classifier(reverse_feature)
        probas = torch.sigmoid(logits)
        return logits, probas, z_log, x_new