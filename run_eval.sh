#!/bin/bash
# path to corsmal dataset without trailing slash (`/`) e.g. folder should have 1/ 2/ 3/ ... 12/ folders
DATA_ROOT="/home/ubuntu/CORSMAL/dataset" # ---- NO TRAILING '/' i.e. not '.....mal/'
# `10` means containers with ids `10`, `11`, `12` etc are going to be used (enough for evaluation)
FIRST_EVAL_CONTAINER=10

# extract features (the first arg `0` is the device id)
cd ./filling_level/vggish
bash ./extract_features.sh 0 $DATA_ROOT $FIRST_EVAL_CONTAINER
cd ../r21d_rgb
bash ./extract_features.sh 0 $DATA_ROOT $FIRST_EVAL_CONTAINER
cd ../../
