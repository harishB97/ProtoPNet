#!/bin/bash

# python local_analysis.py \
#         -modeldir /home/harishbabu/projects/ProtoPNet/saved_models/vgg19/056-cub-bbcrop-lvl1-256-vgg19-10ppc-30ep/ \
#         -model 10_19push0.9051.pth \
#         -imgdir "/home/harishbabu/data/CUB_190_split/official/CUB_200_2011/test_bb_crop_256/195.Carolina_Wren/" \
#         -img Carolina_Wren_0006_186742.jpg \
#         -imgclass 36 \
#         -modellevel 1 \
#         -imglevel 3 \
#         -dataset 'cub'

python local_analysis.py \
        -modeldir /home/harishbabu/projects/ProtoPNet/saved_models/vgg19/064-cub-bbcrop-seg-lvl1-256-vgg19-10ppc-11ep-crcted_mean_std/ \
        -model 10_19push0.9167.pth \
        -imgdir "/home/harishbabu/data/CUB_190_split/official/CUB_200_2011/test_segmented_imagenet_background_bb_crop_256/187.American_Three_toed_Woodpecker/" \
        -img American_Three_Toed_Woodpecker_0045_796148.jpg \
        -imgclass 12 \
        -modellevel 1 \
        -imglevel 3 \
        -dataset 'cub'

exit;


# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt