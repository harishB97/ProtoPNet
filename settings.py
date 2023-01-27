experiment_run = '059-cub-bbcrop-lvl0-256-vgg19-10ppc-20ep-crcted_mean_std'
# experiment_run = '058-fish-pad1-lvl2-256-vgg19_bn-25ppc-20ep-crcted_mean_std'

# model_ckpt_path = '/home/harishbabu/projects/ProtoPNet/saved_models/vgg19/037-fish-pad1-lvl2-256-vgg19-10ppc-30ep/0nopush0.7743.pth'
model_ckpt_path = None

phylo_level = 0

dataset = 'cub'
data_path = '/fastscratch/harishbabu/data/CUB_bb_crop/'
train_dir = data_path + 'train_bb_crop_augmented_256/'
test_dir = data_path + 'test_bb_crop_256/'
train_push_dir = data_path + 'train_bb_crop_256/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75
mean = (0.4671, 0.4643, 0.3998)
std = (0.2372, 0.2332, 0.2567) 

# dataset = 'fish'
# data_path = '/fastscratch/harishbabu/data/Fish/phylo-VQVAE/'
# train_dir = data_path + 'train_global_mean_padded_256_augmented/'
# test_dir = data_path + 'test_global_mean_padded_256/'
# train_push_dir = data_path + 'train_global_mean_padded_256/'
# train_batch_size = 80
# test_batch_size = 100
# train_push_batch_size = 75
# mean = (0.7451, 0.7302, 0.6958)
# std = (0.1604, 0.1922, 0.2335)

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

no_of_prototypes = 25

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

num_train_epochs = 20
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0] + [num_train_epochs-1]

# num_train_epochs = 1
# num_warm_epochs = 5

# push_start = 10
# push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0] + [num_train_epochs-1]

# num_train_epochs = 1
# num_warm_epochs = 0

# push_start = 0
# push_epochs = [0]
