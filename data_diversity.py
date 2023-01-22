import os
src_dir = r'/home/harishbabu/data/Fish/phylo-VQVAE/test_padded_256'

files = [(root.split('/')[-1], len(files)) for i, (root, dirs, files) in enumerate(os.walk(src_dir, topdown = False)) 
                                                if (root != src_dir)]
files = sorted(files, key=lambda x: x[0])

for i, x in enumerate(files):
    print(i, x)
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