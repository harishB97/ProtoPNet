import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
# import torchvision.utils as vutils

import re

import os
import copy
import glob

from helpers import makedir, find_high_activation_crop
import train_and_test as tnt
try:
    from settings import phylo_level
except:
    phylo_level = None

if phylo_level is not None:
    tnt.PHYLO_LEVEL = phylo_level
    tnt.print_phylo_level()
import model
import push
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

activation_percentile = 98

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-imgdir', nargs=1, type=str)
parser.add_argument('-img', nargs=1, type=str)
parser.add_argument('-imgclass', nargs=1, type=int, default=-1)
parser.add_argument('-imglevel', nargs=1, type=int, default=3)
parser.add_argument('-modellevel', nargs=1, type=int, default=3)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# using species level model on ancestor level images
model_level = 2
img_level = 3

import phylogeny_fish as phylo
from collections import defaultdict

descendants = None
species_to_ances = None
if (model_level == 3) and (img_level == 0):
    descendants = defaultdict(list)
    for key, val in phylo.species_to_ances_level0.items():
        descendants[val].append(key)
elif (model_level == 3) and (img_level == 1):
    descendants = defaultdict(list)
    for key, val in phylo.species_to_ances_level1.items():
        descendants[val].append(key)
elif (model_level == 3) and (img_level == 2):
    descendants = defaultdict(list)
    for key, val in phylo.species_to_ances_level2.items():
        descendants[val].append(key)
elif (img_level == 3) and (model_level == 0):
    species_to_ances = phylo.species_to_ances_level2
elif (img_level == 3) and (model_level == 1):
    species_to_ances = phylo.species_to_ances_level2
elif (img_level == 3) and (model_level == 2):
    species_to_ances = phylo.species_to_ances_level2
elif (model_level != img_level):
    raise Exception("Invalid combination of modellevel and imglevel")


# specify the test image to be analyzed
# test_image_dir = args.imgdir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'
# test_image_name = args.img[0] #'Painted_Bunting_0081_15230.jpg'
# test_image_label = 15

test_image_path = r'/home/harishbabu/projects/ProtoPNet/saved_models/vgg19/043-fish-pad1-lvl2-256-vgg19-10ppc-20ep/img/epoch-10/7_prototype-img-original70.png'


# load the model
check_test_accu = False

# load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/'
# load_model_name = args.model[0] #'10_18push0.7822.pth'

#if load_model_dir[-1] == '/':
#    model_base_architecture = load_model_dir.split('/')[-3]
#    experiment_run = load_model_dir.split('/')[-2]
#else:
#    model_base_architecture = load_model_dir.split('/')[-2]
#    experiment_run = load_model_dir.split('/')[-1]

# model_base_architecture = load_model_dir.split('/')[2]
# experiment_run = '/'.join(load_model_dir.split('/')[3:])

# save_analysis_path = os.path.join(test_image_dir, model_base_architecture,
#                                   experiment_run, load_model_name)

# save_analysis_path = os.path.join(load_model_dir, 'local_analysis', load_model_name,
#                                   *test_image_dir.split('/')[-3:], test_image_name.split('.')[0] + '-' + str(activation_percentile))

# makedir(save_analysis_path)

# log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

load_model_path = r'/home/harishbabu/projects/ProtoPNet/saved_models/vgg11/041-fish-pad2-spc-256-vgg11-10ppc-20ep/19nopush0.9057.pth'
# epoch_number_str = re.search(r'\d+', load_model_name).group(0)
# start_epoch_number = int(epoch_number_str)

# log('load model from ' + load_model_path)
# log('model base architecture: ' + model_base_architecture)
# log('experiment run: ' + experiment_run)

print(load_model_path)

ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
# breakpoint()

normalize = transforms.Normalize(mean=mean, std=std)
img_size=256
preprocess = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(), normalize])
load_img = lambda x: Variable(preprocess(Image.open(x).convert('RGB')).unsqueeze(0)).cuda()
train_img = load_img(test_image_path)
feature = ppnet.conv_features(train_img)[0]
# [np.round(feature[:, i, i].detach().cpu().numpy(), 5) for i in range(0, 4)]
# print(np.round(feature[:, 0, 0].detach().cpu().numpy(), 5))

def sat_check():
    saturation = [[] for _ in range(8)]
    max_idx = (0, 0)
    max_saturation = 0
    saturation_count = 0
    for i in range(8):
        for j in range(8):
            saturated = False
            patch = np.round(feature[:, i, j].detach().cpu().numpy(), 5)
            if ((patch == 1.0).sum() + (patch == 0.0).sum()) > 40:
                saturated = True
                saturation_count += 1
            saturation[i].append(saturated)
            if ((patch == 1.0).sum() + (patch == 0.0).sum()) > max_saturation:
                max_saturation = ((patch == 1.0).sum() + (patch == 0.0).sum())
                max_idx = (i, j)

    return saturation, saturation_count, max_idx, max_saturation

saturation, saturation_count, max_idx, max_saturation = sat_check()

print('\nSaturated patches (vectors of shape [128, 1, 1]):')
for row in saturation:
    print(row)
print()
print('Count of saturated patches:', saturation_count, '/64')
print()
i, j = max_idx
print('Patch with maximum saturated values', '(index {', i, '}, {', j, '})')
print(np.round(feature[:, i, j].detach().cpu().numpy(), 5))
print()
print('No of saturated values:', max_saturation, '/128')

# breakpoint()