import os
import numpy as np
from torch.utils.data import Dataset

"""Since our feature extractors are blackbox, instead of loading the images and extracting
    their features, we store the latter into the hard disk and load them"""
    
class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face features"""

    def __init__(self, split = 0):
        # Use the group-balanced validation set for model selection.
        name_dic = {0: 'train', 1:'val', 2:'test'}
        src_dir = '/path/to/features'
        if split == 1:
            print(src_dir)
        self.features = np.load(os.path.join(src_dir, f'{name_dic[split]}_feats.npy'))
        self.bias = np.load(os.path.join(src_dir, f'{name_dic[split]}_bias.npy'))
        self.target = np.load(os.path.join(src_dir, f'{name_dic[split]}_targets.npy'))

    def __getitem__(self, index):
        img = self.features[index]
        label = self.target[index]
        bias = self.bias[index]
        return index, img, label, bias, index

    def __len__(self):
        return self.features.shape[0]


class WaterBirds(Dataset):
    def __init__(self, split):
        src_dir = '/path/to/features'
        if split == 'val':
            split = 'heuristic'
        self.features = np.load(os.path.join(src_dir, f'{split}_feats.npy'))
        self.targets = np.load(os.path.join(src_dir, f'{split}_targets.npy'))
        self.bias = np.load(os.path.join(src_dir, f'{split}_bias.npy'))
        
    def __len__(self):  
        return len(self.features)

    def __getitem__(self, index):
        return index, self.features[index], self.targets[index], self.bias[index], index

    def get_targets(self):   
        return self.targets
    
    def get_biases(self):   
        return self.bias
