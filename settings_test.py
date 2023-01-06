print('-'*25, 'RUNNING TEST', '-'*25)
phylo_level = 0

base_architecture = 'resnet50'
# base_architecture = 'vgg19'
img_size = 256
if phylo_level == 0:
    num_classes = 4
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

experiment_run = 'testrun-fish-256-res50-10ppc-4ep'

data_path = '/home/harishbabu/data/Fish/phylo-VQVAE/'
train_dir = data_path + 'train_padded_256_augmented/'
test_dir = data_path + 'test_padded_256/'
train_push_dir = data_path + 'train_padded_256/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

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

# num_train_epochs = 4
# num_warm_epochs = 1

# push_start = 1
# push_epochs = [i for i in range(num_train_epochs) if i % 2 == 0] + [num_train_epochs-1]

num_train_epochs = 2
num_warm_epochs = 1

push_start = 1
push_epochs = [i for i in range(num_train_epochs) if i % 1 == 0] + [num_train_epochs-1]
