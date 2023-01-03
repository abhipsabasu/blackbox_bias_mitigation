import pandas as pd
import os
from PIL import Image
import utils.config as config
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.create_mnist import create_colored_MNIST


class ColoredMNIST(Dataset):
    def __init__(self, split):
        dataset, labels, colors = create_colored_MNIST(split)
        self.dataset = dataset
        self.labels = labels
        self.colors = colors
        # self.transform = transforms.Compose(
        #     [
        #         transforms.Normalize(0.5, 0.5),
                
        #     ]
        # )
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(config.imagenet_mean, config.imagenet_std),
                
            ]
        )
    def __getitem__(self, index):
        image = self.transform(self.dataset[index].float()  )
        return index, image, self.labels[index], self.colors[index]

    def __len__(self):
        return self.dataset.shape[0]

    

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, split = 0, bias_percentage=None):
        attr = config.bias_attribute
        target = config.target_attribute
        df = None
        if bias_percentage is None:
            attr_df = pd.read_csv(config.attr_path, sep="\s+", skiprows=1, usecols=[target, attr])
            attr_df.loc[attr_df[target] == -1, target] = 0
            attr_df.loc[attr_df[attr] == -1, attr] = 0
            partition_df = pd.read_csv(config.partition_path, sep="\s+", skiprows=0, header=None)
            partition_df.columns = ['Filename', 'Partition']
            partition_df = partition_df.set_index('Filename')
            merge_df = attr_df.merge(partition_df, left_index=True, right_index=True)
            df = merge_df.loc[merge_df['Partition'] == split]
        else:
            if split == 0:
                if bias_percentage == '95':
                    df = pd.read_csv(config.bias95_path, index_col=0)
                elif bias_percentage == '85':
                    df = pd.read_csv(config.bias85_path, index_col=0)
                elif bias_percentage == '75':
                    df = pd.read_csv(config.bias75_path, index_col=0)
                elif bias_percentage == '65':
                    df = pd.read_csv(config.bias65_path, index_col=0)
            elif split == 1:
                df = pd.read_csv(config.balanced_val_path, index_col=0)
            else:
                df = pd.read_csv(config.balanced_test_path, index_col=0)
        self.img_dir = config.img_dir
#         self.csv_path = csv_path
        self.img_names = df.index.values
        self.z = df[attr].values
        self.y = df[target].values
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(config.imagenet_mean, config.imagenet_std),
                
            ]
        )

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        bias = self.z[index]
        return self.img_names[index], img, label, bias

    def __len__(self):
        return self.y.shape[0]
