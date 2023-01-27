#%%
import os
# src_dir = r'/home/harishbabu/data/Fish/phylo-VQVAE/train_padded_256_augmented'
src_dir = r'/fastscratch/harishbabu/data/CUB_bb_crop/train_bb_crop_augmented_256/'
folder_size = [(root.split('/')[-1], len(files)) for i, (root, dirs, files) in enumerate(os.walk(src_dir, topdown = False)) 
                                                if (root != src_dir)]
folder_size = sorted(folder_size, key=lambda x: x[0])
spc_to_img_count = {x: y for x, y in folder_size}
spc_idx_to_img_count = {i: v for i, (k, v) in enumerate(spc_to_img_count.items())}
total_img_count = 0
for folder, size in folder_size:
    total_img_count += size

#%%
# import phylogeny_fish as phylo
import phylogeny_cub as phylo
import matplotlib.pyplot as plt

#%%
# level_mapping = [[x] for x in spc_to_img_count]
level_mapping = phylo.level0_mapping
data_dist = {}
for i, group in enumerate(level_mapping):
    group_img_count = 0
    for spc in group:
        # group_img_count += spc_to_img_count[spc]
        group_img_count += spc_idx_to_img_count[phylo.species_to_idx[spc]]
        total_img_count
    data_dist[i] = group_img_count / total_img_count

print(data_dist)

plt.bar(range(len(data_dist)), list(data_dist.values()),
    tick_label=list(data_dist.keys()))
plt.show()
#%%
# # level_mapping = [[x] for x in spc_to_img_count]
# level_mapping = level2_mapping
# data_dist = {}
# for i, group in enumerate(level_mapping):
#     group_img_count = 0
#     for spc in group:
#         group_img_count += spc_to_img_count[spc]
#     data_dist[i] = group_img_count

# print(data_dist)

# plt.bar(range(len(data_dist)), list(data_dist.values()),
#     tick_label=list(data_dist.keys()))
# plt.show()

#%%
# for i, x in enumerate(folder_size):
#     print(i, x)
    # print(i, root.split('/')[-1], len(files))
    # for name in files:
    #     if (name.endswith(".jpg") or name.endswith(".JPG")):
    #         filename = os.path.join(root, name)
    #         orig_img = Image.open(filename)
    #         target_img = MakeSquared(orig_img, res, mean=global_mean)
    #         target_path = os.path.join(target_dir, os.path.relpath(root, src_dir))
    #         if not os.path.exists(target_path):
    #             os.makedirs(target_path)
    #         target_img.save(os.path.join(target_path, name))
# %%
