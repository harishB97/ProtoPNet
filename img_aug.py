import Augmentor
import os
import time

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

datasets_root_dir = '/fastscratch/mridul/cub_190_split_resized/official/CUB_200_2011/'
dir = datasets_root_dir + 'train_segmented_imagenet_background_bb_crop_256/'
# target_dir = '/home/harishbabu/data/CUB_190_split/official/CUB_200_2011/' + 'train_segmented_imagenet_background_bb_crop_256_augmented/'
target_dir = datasets_root_dir + 'train_segmented_imagenet_background_bb_crop_256_augmented/'  #ORIGINAL
makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

prob_folders = []
start = time.time()
for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]
    # rotation
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    completed = 0
    attempts = 0
    while completed < 10 and attempts < 200:
        try:
            p.process()
            completed += 1
        except:
            prob_folders.append(folders[i])
            # print('*'*30, folders[i])
        attempts += 1

    if completed < 10 and attempts >= 200:
        print('\n', '*^'*30, 'FAILED FOR', folders[i], '\n') 
    del p

    # skew
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    for j in range(10):
        p.process()
    del p
    # shear
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.5)
    for j in range(10):
        p.process()
    del p
    # random_distortion
    #p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    #p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
    #p.flip_left_right(probability=0.5)
    #for j in range(10):
    #    p.process()
    #del p

print((time.time() - start) / 60)
