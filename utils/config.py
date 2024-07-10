# ----------------------Common Hyperparams-------------------------- #
num_class = 1
mlp_neurons = 128

# ----------------------Baseline Hyperparams-------------------------- #
base_epochs = 100
base_batch_size = 256
base_lr = 0.01
weight_decay = 0.01 # Vary this to train a bias-amplified model'

# ----------------------Paths-------------------------- #
basemodel_path = 'basemodel.pth' #'{}_{}_base_balanced.pth'.format(bias_attribute, target_attribute)
margin_path = 'margin.pth' #'{}_{}_adv_balanced.pth'.format(bias_attribute, target_attribute)

# ----------------------Model-details-------------------------- #
model_name = 'resnet18'

# ----------------------ImageNet Means and Transforms---------- #
imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
imagenet_std = [0.229, 0.224, 0.225]

# -----------------------CelebA/Waterbirds-parameters--------- #
dataset_path = '/path/to/dataset'
img_dir = '/path/to/images'
partition_path = '/path/to/dataset_partition'
attr_path = '/path/to/attributes/'
target_attribute = 'Blond_Hair'
bias_attribute = 'Male'
