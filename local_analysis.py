##### MODEL AND DATA LOADING
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
model_level = args.modellevel[0]
img_level = args.imglevel[0]

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
test_image_dir = args.imgdir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'
test_image_name = args.img[0] #'Painted_Bunting_0081_15230.jpg'
test_image_label = args.imgclass[0] #15

test_image_path = os.path.join(test_image_dir, test_image_name)

# load the model
check_test_accu = False

load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/'
load_model_name = args.model[0] #'10_18push0.7822.pth'

#if load_model_dir[-1] == '/':
#    model_base_architecture = load_model_dir.split('/')[-3]
#    experiment_run = load_model_dir.split('/')[-2]
#else:
#    model_base_architecture = load_model_dir.split('/')[-2]
#    experiment_run = load_model_dir.split('/')[-1]

model_base_architecture = load_model_dir.split('/')[2]
experiment_run = '/'.join(load_model_dir.split('/')[3:])

# save_analysis_path = os.path.join(test_image_dir, model_base_architecture,
#                                   experiment_run, load_model_name)

save_analysis_path = os.path.join(load_model_dir, 'local_analysis', load_model_name,
                                  *test_image_dir.split('/')[-3:], test_image_name.split('.')[0] + '-' + str(activation_percentile))

makedir(save_analysis_path)

log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

log('load model from ' + load_model_path)
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run)

ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
# breakpoint()
img_size = ppnet_multi.module.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# load the test data and check test accuracy
from settings import test_dir
if check_test_accu:
    test_batch_size = 100

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    log('test set size: {0}'.format(len(test_loader.dataset)))

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=print)

##### SANITY CHECK
# confirm prototype class identity
load_img_dir = os.path.join(load_model_dir, 'img')

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
prototype_img_identity = prototype_info[:, 5]
prototype_base_img_identity = None
if prototype_info.shape[1] > 6:
    prototype_base_img_identity = prototype_info[:, 6]

log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')

##### HELPER FUNCTIONS FOR PLOTTING
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img

def save_prototype(fname, epoch, index):
    import glob
    file = glob.glob(os.path.join(load_img_dir, 'epoch-'+str(epoch), '*prototype-img'+str(index)+'.png'))[0]
    p_img = plt.imread(file)
    #plt.axis('off')
    plt.imsave(fname, p_img)
    return p_img
    
def save_prototype_self_activation(fname, epoch, index):
    import glob
    file = glob.glob(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    '*prototype-img-original_with_self_act'+str(index)+'.png'))[0]
    p_img = plt.imread(file)
    #plt.axis('off')
    plt.imsave(fname, p_img)
    return p_img

def get_heatmap_from_activation(activation):
    rescaled_act_img = (activation - np.amin(activation))
    rescaled_act_img = rescaled_act_img / np.amax(rescaled_act_img)
    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[...,::-1]
    return heatmap

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    #plt.imshow(img_rgb_float)
    #plt.axis('off')
    plt.imsave(fname, img_rgb_float)
    return img_rgb_float

def save_prototype_original_img_with_bbox(fname, epoch, index, color=(0, 255, 255)):
    # reading activation map
    file = glob.glob(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    '*prototype-self-act'+str(index)+'.npy'))[0]
    proto_act_img = np.load(file)
    upsampled_act_img = cv2.resize(proto_act_img, dsize=(img_size, img_size),
                                             interpolation=cv2.INTER_CUBIC)
    proto_bound, _ = find_high_activation_crop(upsampled_act_img, percentile=activation_percentile)

    # reading original source image of prototype
    file = glob.glob(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    '*prototype-img-original'+str(index)+'.png'))[0]
    orig_img_bgr = cv2.imread(file)
    
    orig_img_bb_bgr = cv2.rectangle(orig_img_bgr, (proto_bound[2], proto_bound[0]), (proto_bound[3]-1, proto_bound[1]-1),
                  color, thickness=2)
    orig_img_bb_rgb = orig_img_bb_bgr[...,::-1]
    orig_img_bb_rgb = np.float32(orig_img_bb_rgb) / 255
    #plt.axis('off')
    plt.imsave(fname, orig_img_bb_rgb)
    return orig_img_bb_rgb

def get_prototype_original_img_with_segmentation(epoch, index):
    # reading activation map
    file = glob.glob(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    '*prototype-self-act'+str(index)+'.npy'))[0]
    proto_act_img = np.load(file)
    upsampled_act_img = cv2.resize(proto_act_img, dsize=(img_size, img_size),
                                             interpolation=cv2.INTER_CUBIC)
    _, activation_mask = find_high_activation_crop(upsampled_act_img, percentile=activation_percentile)
    heatmap_rgb = get_heatmap_from_activation(upsampled_act_img)
    heatmap_bgr = heatmap_rgb[..., ::-1]

    # reading original source image of prototype
    file = glob.glob(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    '*prototype-img-original'+str(index)+'.png'))[0]
    orig_img_bgr = cv2.imread(file)
    # breakpoint()
    # overlayed_original_img_bgr = (0.5 * orig_img_bgr) + (0.3 * ((heatmap_bgr * 255) * np.expand_dims(activation_mask, -1)))
    activation_mask = np.expand_dims(activation_mask, -1)
    segmentation_mask = np.concatenate((np.zeros_like(activation_mask), np.zeros_like(activation_mask), activation_mask),
                                        axis=-1)
    segmentation_mask = (segmentation_mask * 255)
    # segmentation_mask = ((heatmap_bgr * 255) * np.expand_dims(activation_mask, -1))
    # segmentation_mask[:,:,1][segmentation_mask[:,:,1] > 0] = 150
    # segmentation_mask[:,:,0] = 0
    # segmentation_mask[:,:,2] = 0
    overlayed_original_img_bgr = cv2.addWeighted(segmentation_mask.astype(np.float32), 0.2,
                                                    orig_img_bgr.astype(np.float32), 1 - 0.2, 0)
    # breakpoint()
    overlayed_original_img_rgb = overlayed_original_img_bgr[...,::-1]
    overlayed_original_img_rgb = np.float32(overlayed_original_img_rgb) / 255
    return overlayed_original_img_rgb

def save_prototype_original_img_with_segmentation(fname, epoch, index):
    overlayed_original_img_rgb = get_prototype_original_img_with_segmentation(epoch, index, color=(0, 255, 255))
    plt.imsave(fname, overlayed_original_img_rgb)
    return overlayed_original_img_rgb

# def save_prototype_original_img_with_bbox(fname, epoch, index,
#                                           bbox_height_start, bbox_height_end,
#                                           bbox_width_start, bbox_width_end, color=(0, 255, 255)):
#     import glob
#     file = glob.glob(os.path.join(load_img_dir, 'epoch-'+str(epoch), '*prototype-img-original'+str(index)+'.png'))[0]
#     p_img_bgr = cv2.imread(file)
#     if p_img_bgr is None:
#         breakpoint()
#     cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
#                   color, thickness=2)
#     p_img_rgb = p_img_bgr[...,::-1]
#     p_img_rgb = np.float32(p_img_rgb) / 255
#     #plt.imshow(p_img_rgb)
#     #plt.axis('off')
#     plt.imsave(fname, p_img_rgb)
#     return p_img_rgb

# load the test image and forward it through the network
preprocess = transforms.Compose([
   transforms.Resize((img_size,img_size)),
   transforms.ToTensor(),
   normalize
])

img_pil = Image.open(test_image_path)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))

images_test = img_variable.cuda()
labels_test = torch.tensor([test_image_label])

logits, min_distances = ppnet_multi(images_test)
conv_output, distances = ppnet.push_forward(images_test)
prototype_activations = ppnet.distance_2_similarity(min_distances)
prototype_activation_patterns = ppnet.distance_2_similarity(distances)
# stdnorm = lambda x: ((x - x.mean()) / x.std()) # REMOVE
prototype_vectors = ppnet.prototype_vectors.squeeze() # REMOVE
# stdnorm(prototype_vectors[20]) # REMOVE
print([(np.round(prototype_vectors[i].cpu().numpy(), 5) == np.round(prototype_vectors[i+1].cpu().numpy(), 5)) for i in range(0, 25)]) # REMOVE
print('np.round(prototype_vectors[i].cpu().numpy(), 5)')
# breakpoint()
if ppnet.prototype_activation_function == 'linear':
    prototype_activations = prototype_activations + max_dist
    prototype_activation_patterns = prototype_activation_patterns + max_dist

tables = []
for i in range(logits.size(0)):
    tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
    log(str(i) + ' ' + str(tables[-1]))

idx = 0
predicted_cls = tables[idx][0]
correct_cls = tables[idx][1]
log('Predicted: ' + str(predicted_cls))
log('Actual: ' + str(correct_cls))
original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                     images_test, idx)

##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

log('Most activated 10 prototypes of this image:')
most_activated_df = pd.DataFrame()
top_k_imgs = []
array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
for i in range(1,11):
    log('top {0} activated prototype for this image:'.format(i))
    most_activated_dict = {}
    save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'top-%d_activated_prototype.png' % i),
                   start_epoch_number, sorted_indices_act[-i].item())
    orig_bb = save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                             'top-%d_activated_prototype_in_original_pimg.png' % i),
                                          epoch=start_epoch_number,
                                          index=sorted_indices_act[-i].item(),
                                        #   bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                        #   bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                        #   bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                        #   bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                          color=(0, 255, 255))
    orig_self_act = save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                'top-%d_activated_prototype_self_act.png' % i),
                                   start_epoch_number, sorted_indices_act[-i].item())[...,:-1]

    orig_segmentation = get_prototype_original_img_with_segmentation(start_epoch_number, sorted_indices_act[-i].item())

    log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
    log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
    if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
        log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
    log('activation value (similarity score): {0}'.format(array_act[-i]))
    log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))

    most_activated_df = most_activated_df.append({'prototype index':sorted_indices_act[-i].item(),
                                                    'prototype class identity': prototype_img_identity[sorted_indices_act[-i].item()],
                                                    'similarity score': array_act[-i].cpu().item()}, ignore_index=True)
    try:
        most_activated_df.to_csv(os.path.join(save_analysis_path, 'most_activated_prototypes', 'most_activated.csv'))
    except:
        print('Unable to save most_activated.csv')
    
    activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
    upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                              interpolation=cv2.INTER_CUBIC)
    
    # show the most highly activated patch of the image by this prototype
    high_act_patch_indices, activation_mask = find_high_activation_crop(upsampled_activation_pattern, percentile=activation_percentile)
    high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                  high_act_patch_indices[2]:high_act_patch_indices[3], :]
    log('most highly activated patch of the chosen image by this prototype:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'most_highly_activated_patch_by_top-%d_prototype.png' % i),
               high_act_patch)
    log('most highly activated patch by this prototype shown in the original image:')
    img_bb = imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                     img_rgb=original_img,
                     bbox_height_start=high_act_patch_indices[0],
                     bbox_height_end=high_act_patch_indices[1],
                     bbox_width_start=high_act_patch_indices[2],
                     bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
    
    # show the image overlayed with prototype activation map
    rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
    rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[...,::-1]
    
    activation_mask = np.expand_dims(activation_mask, -1)
    segmentation_mask = np.concatenate((np.zeros_like(activation_mask), np.zeros_like(activation_mask), activation_mask),
                                        axis=-1)
    segmentation_mask = (segmentation_mask * 255)
    # img_segmentation = (0.5 * original_img) + (0.3 * (heatmap * np.expand_dims(activation_mask, -1)))
    img_segmentation = cv2.addWeighted(segmentation_mask.astype(np.float32), 0.2,
                                                    (original_img[...,::-1]*255).astype(np.float32), 1 - 0.2, 0)
    img_segmentation = (img_segmentation[...,::-1] / 255).astype(np.float32)
    overlayed_img = 0.5 * original_img + 0.3 * heatmap
    # breakpoint()
    log('prototype activation map of the chosen image:')
    #plt.axis('off')
    img_act = overlayed_img
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'prototype_activation_map_by_top-%d_prototype.png' % i),
               overlayed_img)
    log('--------------------------------------------------------------')

    # row = torch.stack([torch.tensor(img_bb), torch.tensor(orig_bb), torch.tensor(img_act), torch.tensor(orig_self_act[...,:-1])])
    # breakpoint()
    # vutils.save_image(row, 
    #                     os.path.join(save_analysis_path, 'most_activated_prototypes', 
    #                             'top-%d_prototype.png' % i), 
    #                     padding=4)
    img_bb = cv2.copyMakeBorder(img_bb, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
    orig_bb = cv2.copyMakeBorder(orig_bb, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
    img_act = cv2.copyMakeBorder(img_act, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
    orig_self_act = cv2.copyMakeBorder(orig_self_act, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
    orig_segmentation = cv2.copyMakeBorder(orig_segmentation, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
    img_segmentation = cv2.copyMakeBorder(img_segmentation, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
    row = np.hstack([img_bb, orig_bb, img_segmentation, orig_segmentation, img_act, orig_self_act]).astype(np.float32)
    row = cv2.cvtColor(row*255, cv2.COLOR_RGB2BGR)
    proto_class_id = prototype_img_identity[sorted_indices_act[-i].item()]
    proto_base_class_id = None
    if prototype_base_img_identity is not None:
        proto_base_class_id = prototype_base_img_identity[sorted_indices_act[-i].item()]
    if descendants:
        if (proto_class_id in descendants[test_image_label]):
            text_color = (4, 143, 14)
        else:
            text_color = (0, 0, 255)
    if species_to_ances:
        if proto_class_id == species_to_ances[test_image_label]:
            text_color = (4, 143, 14)
        else:
            text_color = (0, 0, 255)
    elif proto_class_id == test_image_label:
        text_color = (4, 143, 14)
    else:
        text_color = (0, 0, 255)

    # text = " ".join(['PROTO CLASS:', str(proto_class_id),
    #                 'SCORE:', str(round(array_act[-i].item(), 2))])
    text = 'PROTO CLASS: ' + str(proto_class_id)
    if proto_base_class_id:
        text += " | " + 'PROTO ORIGIN SPC: ' + str(proto_base_class_id)
    text += " | " + 'SCORE: ' + str(round(array_act[-i].item(), 2))
    text += " | " + 'THRSH: ' + str(activation_percentile)
    row = cv2.putText(row, text, org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75, color=text_color, thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(save_analysis_path, 'most_activated_prototypes', 
                                'comparison_top-%d_prototype.png' % i),
               row)
    top_k_imgs.append(row)

    # breakpoint()


# vutils.save_image(torch.vstack(top_k_imgs), 
#                         os.path.join(save_analysis_path, 'most_activated_prototypes', 
#                                 'consolidated_top-10_prototypes.png' % i), 
#                         padding=4)
# top_k_imgs = cv2.cvtColor(np.vstack(top_k_imgs), cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(save_analysis_path, 'most_activated_prototypes', 
                                'comparison_consolidated_top-10_prototypes.png'),
               np.vstack(top_k_imgs))

# breakpoint()



# ##### PROTOTYPES FROM TOP-k CLASSES
# k = 5
# log('Prototypes from top-%d classes:' % k)
# prototype_top_k_classes_df = pd.DataFrame()
# topk_logits, topk_classes = torch.topk(logits[idx], k=k)
# for i,c in enumerate(topk_classes.detach().cpu().numpy()):
#     makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))

#     log('top %d predicted class: %d' % (i+1, c))
#     log('logit of the class: %f' % topk_logits[i])
#     class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
#     class_prototype_activations = prototype_activations[idx][class_prototype_indices]
#     _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

#     prototype_cnt = 1
#     for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
#         prototype_index = class_prototype_indices[j]
#         save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
#                                     'top-%d_activated_prototype.png' % prototype_cnt),
#                        start_epoch_number, prototype_index)
#         save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
#                                                                  'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
#                                               epoch=start_epoch_number,
#                                               index=prototype_index,
#                                               bbox_height_start=prototype_info[prototype_index][1],
#                                               bbox_height_end=prototype_info[prototype_index][2],
#                                               bbox_width_start=prototype_info[prototype_index][3],
#                                               bbox_width_end=prototype_info[prototype_index][4],
#                                               color=(0, 255, 255))
#         save_prototype_self_activation(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
#                                                     'top-%d_activated_prototype_self_act.png' % prototype_cnt),
#                                        start_epoch_number, prototype_index)
#         log('prototype index: {0}'.format(prototype_index))
#         log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
#         if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
#             log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
#         log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
#         log('last layer connection: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))

#         prototype_top_k_classes_df = prototype_top_k_classes_df.append({'prototype index':prototype_index,
#                                                     'prototype class identity': prototype_img_identity[prototype_index],
#                                                     'similarity score': prototype_activations[idx][prototype_index].cpu().item()},
#                                                      ignore_index=True)
#         try:
#             prototype_top_k_classes_df.to_csv(os.path.join(save_analysis_path, 'most_activated_prototypes', 'prototype_top_k_classes.csv'))
#         except:
#             print('Unable to save most_activated.csv')
        
#         activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
#         upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
#                                                   interpolation=cv2.INTER_CUBIC)
        
#         # show the most highly activated patch of the image by this prototype
#         high_act_patch_indices, _ = find_high_activation_crop(upsampled_activation_pattern, percentile=activation_percentile)
#         high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
#                                       high_act_patch_indices[2]:high_act_patch_indices[3], :]
#         log('most highly activated patch of the chosen image by this prototype:')
#         #plt.axis('off')
#         plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
#                                 'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
#                    high_act_patch)
#         log('most highly activated patch by this prototype shown in the original image:')
#         imsave_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
#                                             'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
#                          img_rgb=original_img,
#                          bbox_height_start=high_act_patch_indices[0],
#                          bbox_height_end=high_act_patch_indices[1],
#                          bbox_width_start=high_act_patch_indices[2],
#                          bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
#         # show the image overlayed with prototype activation map
#         rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
#         rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
#         heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
#         heatmap = np.float32(heatmap) / 255
#         heatmap = heatmap[...,::-1]
#         overlayed_img = 0.5 * original_img + 0.3 * heatmap
#         log('prototype activation map of the chosen image:')
#         #plt.axis('off')
#         plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
#                                 'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
#                    overlayed_img)
#         log('--------------------------------------------------------------')
#         prototype_cnt += 1
#     log('***************************************************************')

# if predicted_cls == correct_cls:
#     log('Prediction is correct.')
# else:
#     log('Prediction is wrong.')

# logclose()

# # breakpoint()

