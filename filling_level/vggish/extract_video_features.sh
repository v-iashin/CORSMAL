#!/bin/bash
# you may want to switch it to `anaconda3` if you install anaconda instead of miniconda
source ~/miniconda3/etc/profile.d/conda.sh

# path to corsmal dataset without trailing slash (`/`) e.g. folder should have 1/ 2/ 3/ ... 12/ folders
DATA_ROOT="/home/nvme/vladimir/corsmal" # ---- NO TRAILING '/' i.e. not '.....mal/'

# making sure we are not in 'base' or any other
conda deactivate
conda deactivate

# make sure to install it first (follow the guidelines in `./video_features` folder in README.md)
# make sure to download the model
conda activate vggish

# form paths for feature extraction (will be saved to ./filepaths)
python ./form_filenames.py --data_root $DATA_ROOT

# moving to `./video_features` folder
cd ./video_features

for container_id in {1..12}
do
/home/vladimir/miniconda3/envs/vggish/bin/python main.py \
    --feature_type vggish \
    --device_ids 0 1 \
    --on_extraction save_numpy \
    --file_with_video_paths "../filepaths/"$container_id"_audio_file_paths.txt" \
    --output_path "../vggish_features/"$container_id"/vggish"
done
