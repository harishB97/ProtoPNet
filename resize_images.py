
import pandas as pd
import numpy as np
import os
import cv2
from multiprocessing import Lock, Pool
import time

n_threads = 4
IMG_DIR = r'/home/harishbabu/data/CUB_190_split/official/CUB_200_2011/train_bb_crop'
SAVE_PATH = r'/home/harishbabu/data/CUB_190_split/official/CUB_200_2011/train_bb_crop_256'
folders = [os.path.join(IMG_DIR, folder) for folder in next(os.walk(IMG_DIR))[1]]
target_folders = [os.path.join(SAVE_PATH, folder) for folder in next(os.walk(IMG_DIR))[1]]

# # ALL_IMG_PATHS = os.listdir(IMG_DIR)#[:200]
# ALL_IMG_PATHS = folders
# print(len(ALL_IMG_PATHS))
# n = len(ALL_IMG_PATHS) // n_threads
# ALL_IMG_PATHS = [ALL_IMG_PATHS[i * n:(i + 1) * n] for i in range((len(ALL_IMG_PATHS) + n - 1) // n )]
# while len(ALL_IMG_PATHS) > n_threads:
#     ALL_IMG_PATHS[-2] += ALL_IMG_PATHS[-1]
#     ALL_IMG_PATHS = ALL_IMG_PATHS[:-1]
# print(len(ALL_IMG_PATHS[-1]))
# print(len(ALL_IMG_PATHS))

def split_for_threads(folders):
    print(len(folders))
    n = len(folders) // n_threads
    folders = [folders[i * n:(i + 1) * n] for i in range((len(folders) + n - 1) // n )]
    while len(folders) > n_threads:
        folders[-2] += folders[-1]
        folders = folders[:-1]
    print(len(folders[-1]))
    print(len(folders))

    return folders

folders = split_for_threads(folders)
target_folders = split_for_threads(target_folders)

class ResizeThread():
    
    def __init__(self, thread_id, folders, target_folders, lock):
        self.thread_id = thread_id
        self.folders = folders
        self.target_folders = target_folders
        self.lock = lock
        
    def resize_images(self):
        time_ = time.time()
        print("Started", self.thread_id, time_)
        j = 0
        for read_folder, write_folder in zip(self.folders, self.target_folders):
            os.makedirs(write_folder, exist_ok=True)
            for filename in os.listdir(read_folder):
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    file = os.path.join(read_folder, filename)
                    img = cv2.imread(file)
                    img = cv2.resize(img, (256, 256))
                    cv2.imwrite(os.path.join(write_folder, filename), img)
                    j += 1
                    if j%500 == 0:
                        self.lock.acquire()
                        print('Process:', self.thread_id, 'completed', j, '. Time', time.time()-time_)
                        self.lock.release()


resize_threads = []
lock = Lock()
for t in range(n_threads):
    resize_threads.append(ResizeThread(t+1, folders[t], target_folders[t], lock))

def resize_images(i):
    resize_threads[i].resize_images()
    
if __name__ == '__main__':
    lock = Lock()
    with Pool() as p:
            print(p.map(resize_images, list(range(n_threads))))
