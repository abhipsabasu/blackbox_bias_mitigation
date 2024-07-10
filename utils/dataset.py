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
        src_dir = '/data2/abhipsa/blackbox_bias_mitigation/blackbox-codebase/waterbirds_embeddings/'
        if split == 'val':
            src_dir = '/data2/abhipsa/blackbox_bias_mitigation/blackbox-codebase/waterbirds_embeddings/all_balanced'
        if split == 'train':
            
            # indices = np.load('/data2/abhipsa/blackbox_bias_mitigation/blackbox-codebase/3000_wb.npy')
            self.features = np.load(
                os.path.join(src_dir, f"{split}_feats.npy")
            ).astype(np.float32) #[indices]
            self.targets = np.load(os.path.join(src_dir, f"{split}_targets.npy")) #[indices]
            self.bias = np.load(os.path.join(src_dir, f"{split}_bias.npy")) #[indices]
            self.class_sample_count = np.array(
            [len(np.where(self.targets == t)[0]) for t in np.unique(self.targets)])
        else:
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

class WaterBirdsActual(Dataset):
    def __init__(self, split):
        src_dir = '/data2/abhipsa/blackbox_bias_mitigation/blackbox-codebase/sup_embeddings/actual_images/waterbirds_embeddings'
        # if split == 'val':
        #     src_dir = src_dir = '/data2/abhipsa/blackbox_bias_mitigation/blackbox-codebase/waterbirds_embeddings/all_balanced'
        self.features = np.load(os.path.join(src_dir, f'{split}_images.npy'))
        self.targets = np.load(os.path.join(src_dir, f'{split}_targets.npy'))
        self.bias = np.load(os.path.join(src_dir, f'{split}_bias.npy'))
        if split == 'train':
            self.class_sample_count = np.array(
            [len(np.where(self.targets == t)[0]) for t in np.unique(self.targets)])
        if split == 'val':
            all_bias = np.unique(self.bias)
            all_targets = np.unique(self.targets)
            groups = [self.bias[(self.bias==i) & (self.targets==j)].shape[0] for i in all_bias for j in all_targets]
            min_groups_reqd = min(groups)
            group_indices = [(i, j) for i in all_bias for j in all_targets]
            all_indices = [np.arange(groups[0]), np.arange(groups[1]), np.arange(groups[2]), np.arange(groups[3])]
            all_indices = [np.random.choice(indices, min_groups_reqd) for indices in all_indices]
            # print(all_indices)
            # g1_indices, g2_indices, g3_indices, g4_indices = np.random.choice(g1_indices), np.random.choice(g2_indices), np.random.choice(g3_indices), np.random.choice(g4_indices)
            # print(all_indices)
            print(type(self.features))
            self.features = np.array([self.features[(self.bias==group_indices[e][0]) & (self.targets==group_indices[e][1])][all_indices[e]] for e in range(len(all_indices))]).reshape(-1, 3, 224, 224)
            bias = np.array([self.bias[(self.bias==group_indices[e][0]) & (self.targets==group_indices[e][1])][all_indices[e]] for e in range(len(all_indices))]).reshape(-1)
            targets = np.array([self.targets[(self.bias==group_indices[e][0]) & (self.targets==group_indices[e][1])][all_indices[e]] for e in range(len(all_indices))]).reshape(-1)
            self.bias = bias
            self.targets = targets
            print(self.features.shape, self.bias.shape, self.targets.shape, type(self.features))
        # print(self.features.shape)
        # self.features = torch.from_numpy(self.features)
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.Normalize(
            #     [0.485, 0.456, 0.406], 
            #     [0.229, 0.224, 0.225]
            # )
        ])
        train_transform = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
            
            # transforms.Normalize(
            #     [0.485, 0.456, 0.406], 
            #     [0.229, 0.224, 0.225]
            # )
        ])
        if split == 'train':
            self.transform = train_transform
        else:
            self.transform = transform
        
    def __len__(self):  
        return len(self.features)

    def __getitem__(self, index):
        return index, self.features[index], self.targets[index], self.bias[index], index

    def get_targets(self):   
        return self.targets
    
    def get_biases(self):   
        return self.bias


