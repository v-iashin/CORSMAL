#!/bin/bash
DEVICE=$1
# the same as =$2 but removes trailing slash in the path
DATA_ROOT=${2%/}

# you may want to switch it to `anaconda3` if you install anaconda instead of miniconda
source ~/miniconda3/etc/profile.d/conda.sh

# making sure we are not in 'base' or any other
conda deactivate
conda deactivate
conda activate vggish

# change dir to `./video_features` folder
# also patching the video_features code by creating tmp dir otherwise will fail with system error
cd ./video_features && mkdir ./tmp

for container_dir in *$DATA_ROOT/
do
# take only the last folder in the path which is the container id
container_id="$(basename $container_dir)"

# run feature extraction
python main.py \
    --feature_type vggish \
    --device_ids $DEVICE \
    --on_extraction save_numpy \
    --video_paths $(find $DATA_ROOT/$container_id/audio -name "*.wav")\
    --output_path "../vggish_features/"$container_id"/vggish"
done
