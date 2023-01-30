import glob
from helpers import find_high_activation_crop
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

load_model_dir = r'/home/harishbabu/projects/ProtoPNet/saved_models/vgg19_bn/067-fish-pad1-spc-256-vgg19_bn-10ppc-11ep-crcted_mean_std-bn_before_sigmoid/'
load_img_dir = os.path.join(load_model_dir, 'img')
img_size = 256
activation_percentile = 98 # 95 for CUB

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


def main():
    bb_prototypes_dir = os.path.join(load_model_dir, 'bb_prototypes')
    os.makedirs(bb_prototypes_dir, exist_ok=True)

    epoch = 10
    bb_dir = os.path.join(load_img_dir, 'epoch-'+str(epoch), 'bb'+str(epoch) + '.npy')
    bb = np.load(bb_dir)
    no_of_prototypes = bb.shape[0]
    # print(bb.shape)
    for i in range(no_of_prototypes):
        original_label = bb[i, 5]
        species_label = bb[i, 6]
        if 'spc' in load_model_dir:
            os.makedirs(os.path.join(bb_prototypes_dir, 'spc'+str(original_label)), exist_ok=True)
            fname = os.path.join(bb_prototypes_dir, 'spc'+str(original_label), 'spc'+str(species_label)+'_prototype_bb_'+str(i)+ '.png')
        else:
            os.makedirs(os.path.join(bb_prototypes_dir, 'ances'+str(original_label)), exist_ok=True)
            fname = os.path.join(bb_prototypes_dir, 'ances'+str(original_label), 'spc'+str(species_label)+'_prototype_bb_'+str(i)+ '.png')
        save_prototype_original_img_with_bbox(fname, epoch, i, color=(0, 255, 255))

if __name__ == '__main__':
    main()