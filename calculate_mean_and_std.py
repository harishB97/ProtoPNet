import numpy as np
import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from settings import train_batch_size
import time

# train_dir = '/fastscratch/harishbabu/data/Fish/phylo-VQVAE/test_global_mean_padded_256'
# train_dir = '/fastscratch/harishbabu/data/CUB_bb_crop/train_bb_crop_augmented_256/'
train_dir = '/home/harishbabu/data/CUB_190_split/official/CUB_200_2011/train_segmented_imagenet_background_bb_crop_256_augmented'
img_size = 256

print(train_dir)

train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        # normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)

dataloader = train_loader

# Calculating mean
start = time.time()
sum_pixel_values = [0, 0, 0] #RGB
pixel_count = 0
# print(sum_pixel_values.dtype)
for i, (image, label) in enumerate(dataloader):
    for j in range(3):
        sum_pixel_values[j] += image[:, j, :, :].sum()
    pixel_count += (image.shape[-1] * image.shape[-2] * image.shape[0])

mean = [0]*3
for i in range(3):
    mean[i] = sum_pixel_values[i] / pixel_count
print(mean)
print(time.time() - start)

# Calculating standard deviation
start = time.time()
std = [0]*3
pixel_count = 0
sum_squre_diff = [0]*3
for i, (image, label) in enumerate(dataloader):
    for j in range(3):
        sum_squre_diff[j] += torch.square(image[:, j, :, :] - mean[j]).sum()
    pixel_count += (image.shape[-1] * image.shape[-2] * image.shape[0])

for i in range(3):
    std[i] = torch.sqrt(sum_squre_diff[i] / pixel_count)
print(std)
print(time.time() - start)

"""
train_dir = '/fastscratch/harishbabu/data/Fish/phylo-VQVAE/train_global_mean_padded_256_augmented'
    mean: [tensor(0.7451), tensor(0.7302), tensor(0.6958)]
    std: [tensor(0.1604), tensor(0.1922), tensor(0.2335)]

train_dir = '/fastscratch/harishbabu/data/Fish/phylo-VQVAE/test_global_mean_padded_256'
    mean: [tensor(0.7486), tensor(0.7361), tensor(0.7063)]
    std: [tensor(0.1503), tensor(0.1792), tensor(0.2174)]

train_dir = '/fastscratch/harishbabu/data/CUB_bb_crop/train_bb_crop_augmented_256/'
    mean: [tensor(0.4671), tensor(0.4643), tensor(0.3998)]
    std: [tensor(0.2372), tensor(0.2332), tensor(0.2567)]

train_dir = '/fastscratch/mridul/cub_190_split_resized/official/CUB_200_2011/train_segmented_imagenet_background_bb_crop_256'
    mean: [tensor(0.4728), tensor(0.4410), tensor(0.3919)]
    std: [tensor(0.1265), tensor(0.1201), tensor(0.1221)]

train_dir = '/home/harishbabu/data/CUB_190_split/official/CUB_200_2011/train_segmented_imagenet_background_bb_crop_256_augmented'
    mean: [tensor(0.4699), tensor(0.4381), tensor(0.3879)]
    std: [tensor(0.1360), tensor(0.1292), tensor(0.1311)]
"""