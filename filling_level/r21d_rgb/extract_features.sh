#!/bin/bash
DEVICE=$1
# the same as =$2 but removes trailing slash in the path
DATA_ROOT=${2%/}

# you may want to switch it to `anaconda3` if you install anaconda instead of miniconda
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

# making sure we are not in 'base' or any other
conda deactivate
conda deactivate
conda activate r21d

# change dir to `./video_features` folder
# also patching the video_features code by creating tmp dir otherwise will fail with system error
cd ./video_features && mkdir -p ./tmp

for container_dir in $DATA_ROOT/*
do
# take only the last folder in the path which is the container id
container_id="$(basename $container_dir)"

# run feature extraction
python main.py \
    --feature_type r21d_rgb \
    --device_ids $DEVICE \
    --on_extraction save_numpy \
    --video_paths $(find $DATA_ROOT/$container_id/rgb/ -name "*.mp4")\
    --output_path "../r21d_rgb_features/"$container_id"/r21d_rgb"
done
