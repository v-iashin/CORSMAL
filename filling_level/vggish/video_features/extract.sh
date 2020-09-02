#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda deactivate

for value in {10..12}
do

# VGGish
# conda activate vggish
# /home/vladimir/miniconda3/envs/vggish/bin/python main.py \
#     --feature_type vggish \
#     --device_ids 0 1 \
#     --on_extraction save_numpy \
#     --file_with_video_paths "/home/vladimir/ISSS/filepaths/"$value"_audio_file_paths.txt" \
#     --output_path "/home/nvme/vladimir/corsmal/features/"$value"/vggish"
# conda deactivate
# conda deactivate

# R(2+1)d
conda activate r21d
/home/vladimir/miniconda3/envs/r21d/bin/python main.py \
    --feature_type r21d_rgb \
    --device_ids 0 1 \
    --on_extraction save_numpy \
    --file_with_video_paths "/home/vladimir/ISSS/filepaths/"$value"_rgb_file_paths.txt" \
    --output_path "/home/nvme/vladimir/corsmal/features/"$value"/r21d_rgb"
conda deactivate
conda deactivate
done
