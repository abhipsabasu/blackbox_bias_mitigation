# ----------------------Common Hyperparams-------------------------- #
num_class = 1
target_attribute = 'Smiling'
bias_attribute = 'Male'
mlp_neurons = 128

# ----------------------Baseline Hyperparams-------------------------- #
base_epochs = 30
base_batch_size = 32
base_lr = 1e-4
weight_decay = 0

# ----------------------Adversarial Hyperparams-------------------------- #
adv_epochs = 30
adv_batch_size = 32
adv_lr = 5e-4
adv_weight_decay = 1e-5
lambda_cluster = 1e+6
lambda_adv = 1.0
take_c1 = True

# ----------------------Paths-------------------------- #
img_dir = '../datasets/celeba/img_align_celeba/'
partition_path = '../datasets/celeba/list_eval_partition.txt'
attr_path = '../datasets/celeba/list_attr_celeba.txt'
basemodel_path = '{}_{}_base_balanced.pth'.format(bias_attribute, target_attribute)
adv_path = '{}_{}_adv_balanced.pth'.format(bias_attribute, target_attribute)
bias95_path = '../datasets/celeba/celeba-gender-partitions_train_0.95_0.95.csv'
bias85_path = '../datasets/celeba/celeba-gender-partitions_train_0.85_0.85.csv'
bias75_path = '../datasets/celeba/celeba-gender-partitions_train_0.75_0.75.csv'
bias65_path = '../datasets/celeba/celeba-gender-partitions_train_0.65_0.65.csv'
balanced_val_path = '../datasets/celeba/celeba-gender-partitions_bal_val.csv'
balanced_test_path = '../datasets/celeba/celeba-gender-partitions_bal_test.csv'

# ----------------------Model-details-------------------------- #
model_name = 'resnet18'

# ----------------------ImageNet Means and Transforms---------- #
imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
imagenet_std = [0.229, 0.224, 0.225]

# ----------------------MNIST-parameters---------- #
class1 = 0
class2 = 8
left1 = 4
right1 = 0
left2 = 2
right2 = 3
mnist_bias = 1
color_path = '/data2/abhipsa/DebiAN/datasets/make_datasets/colors.th'
MNIST_dir = '../datasets/MNIST'
colored_mnist_dir = '../datasets/MNIST/colored_mnist'
split = 0.9

