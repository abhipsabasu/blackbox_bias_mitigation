import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import utils.config as config


class Network(nn.Module):
    def __init__(self, model_name, num_classes):
        super(Network, self).__init__()
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
                        nn.ReLU(),
                )
        self.classifier = nn.Linear(config.mlp_neurons, self.num_classes)

    def forward(self, x):
        x = self.feats(x).squeeze()
        x = self.new_feats(x)
        logits = self.classifier(x)
        probas = torch.sigmoid(logits)
        return logits, probas, x 