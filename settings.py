experiment_run = '051-cub-bbcrop-spc-256-vgg19-10ppc-60ep'

# model_ckpt_path = '/home/harishbabu/projects/ProtoPNet/saved_models/vgg19/036-fish-pad1-lvl0-256-vgg19-10ppc-30ep/9nopush0.9722.pth'
model_ckpt_path = None

phylo_level = None

dataset = 'cub'
data_path = '/home/harishbabu/data/CUB_190_split/official/CUB_200_2011/'
train_dir = data_path + 'train_bb_crop_augmented_256/'
test_dir = data_path + 'test_bb_crop_256/'
train_push_dir = data_path + 'train_bb_crop_256/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

# dataset = 'fish'
# data_path = '/fastscratch/harishbabu/data/Fish/phylo-VQVAE/'
# train_dir = data_path + 'train_global_mean_padded_256_augmented/'
# test_dir = data_path + 'test_global_mean_padded_256/'
# train_push_dir = data_path + 'train_global_mean_padded_256/'
# train_batch_size = 80
# test_batch_size = 100
# train_push_batch_size = 75

img_size = 256

# base_architecture = 'resnet18'
base_architecture = 'vgg19'

if dataset == 'cub':
    if phylo_level == 0:
        num_classes = 16
    elif phylo_level == 1:
        num_classes = 37
    elif phylo_level == 2:
        num_classes = 85
    else:
        num_classes = 190
else:
    if phylo_level == 0:
        num_classes = 3
    elif phylo_level == 1:
        num_classes = 6
    elif phylo_level == 2:
        num_classes = 9
    else:
        num_classes = 38

no_of_prototypes = 10

prototype_shape = (num_classes*no_of_prototypes, 128, 1, 1) # (No. of prototypes, channels, height, width)

prototype_activation_function = 'log'
add_on_layers_type = 'regular'


joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 60
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0] + [num_train_epochs-1]

# num_train_epochs = 1
# num_warm_epochs = 0

# push_start = 0
# push_epochs = [0]
