#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=1-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ./SLURM/slurm-%j.out


echo start load env and run python

module reset
module load Anaconda3/2020.11
# module load gcc/8.2.0
# module reset
# module load Anaconda3/2020.11
# TODO: there is a bug. for some reason I need to reset again here.
source activate taming3
module reset
source activate taming3
which python

python run_pruning.py -modeldir /home/harishbabu/projects/ProtoPNet/saved_models/vgg19/004-fish-256-vgg19-10ppc-20ep/ -model 10_19push0.9235.pth

# python global_analysis.py -modeldir "/home/harishbabu/projects/ProtoPNet/saved_models/vgg19/002-fish-3ppc-20ep" -model "/home/harishbabu/projects/ProtoPNet/saved_models/vgg19/002-fish-3ppc-20ep/10nopush0.9127.pth"

exit;




# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt