# ----------------------Common Hyperparams-------------------------- #
num_class = 1
mlp_neurons = 128

# ----------------------Baseline Hyperparams-------------------------- #
base_epochs = 50
base_batch_size = 512
base_lr = 0.0001
weight_decay = 0.1 # Vary this to train a bias-amplified model'
scale = 8
std = 0.15
K = 2

opt_b = 'sgd'
opt_m = 'sgd'

hid_dim = 1024
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
celeba_path = '/data2/abhipsa/blackbox_bias_mitigation/blackbox-codebase/sup_embeddings/clip/celeba_embeddings/Blond_Hair'
celeba_val_path = '/data2/abhipsa/blackbox_bias_mitigation/blackbox-codebase/sup_embeddings/clip/celeba_embeddings/Blond_Hair/all_balanced'
waterbirds_path = '/path/to/waterbirds_traintest_embeddings'
waterbirds_val_path = '/path/to/waterbirds_val_balanced_embeddings'
