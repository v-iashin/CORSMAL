#!/bin/bash
# path to corsmal dataset without trailing slash (`/`) e.g. folder should have 1/ 2/ 3/ ... 12/ folders
DATA_ROOT="/home/ubuntu/CORSMAL/dataset" # ---- NO TRAILING '/' i.e. not '.....mal/'

# you may want to switch it to `anaconda3` if you install anaconda instead of miniconda
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

echo "STARTING PART 1/3..."
# `10` means containers with ids `10`, `11`, `12` etc are going to be used (enough for evaluation)
FIRST_EVAL_CONTAINER=1
# extract features (the first arg `0` is the device id)
cd ./filling_level/vggish
# bash ./extract_features.sh 0 $DATA_ROOT $FIRST_EVAL_CONTAINER
# cd ../r21d_rgb
# bash ./extract_features.sh 0 $DATA_ROOT $FIRST_EVAL_CONTAINER
cd ../../

# making sure we are not in 'base' or any other env
conda deactivate
conda activate corsmal

# Filling level VGGish
cd ./filling_level/vggish
# remove `--predict_on_private` if you don't have it and it will make predictions only for public test set
python main.py --predict_on_private
cd ../../

# Filling Type VGGish
cd ./filling_type/vggish
# remove `--predict_on_private` if you don't have it and it will make predictions only for public test set
python main.py --predict_on_private
cd ../../

conda deactivate
