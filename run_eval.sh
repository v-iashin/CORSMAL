#!/bin/bash
# path to corsmal dataset without trailing slash (`/`) e.g. folder should have 1/ 2/ 3/ ... 12/ folders
DATA_ROOT="/home/ubuntu/CORSMAL/dataset" # ---- NO TRAILING '/' i.e. not '.....mal/'

# you may want to switch it to `anaconda3` if you install anaconda instead of miniconda
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

echo "STARTING PART 1/3..."
# `10` means containers with ids `10`, `11`, `12` etc are going to be used (enough for evaluation)
# if `1` then all containers will be used
FIRST_EVAL_CONTAINER=1
# extract features (the first arg "0 1 2 3" is the device id)
cd ./filling_level/vggish
bash ./extract_features.sh "0 1 2 3" $DATA_ROOT $FIRST_EVAL_CONTAINER
# cd ../r21d_rgb
# bash ./extract_features.sh "0 1 2 3" $DATA_ROOT $FIRST_EVAL_CONTAINER
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

# echo "STARTING PART 2/3..."
# conda activate LoDE
# cd ./capacity
# # prediction on private dataset (expected to be 13, 14, 15 containers)
# python main.py --data_path $DATA_ROOT --predict_on_private
# # prediction on public dataset (expected to be 10, 11, 12)
# python main.py --data_path $DATA_ROOT
# cd ../

# conda deactivate

# echo "STARTING PART 3/3..."
# conda activate pyAudioAnalysis
# # filling_level
# cd ./filling_level/CORSMAL-pyAudioAnalysis/
# chmod +x ./gather_final_dataset.sh
# # retriveing the relative path to the dataset from this folder
# source_data_path=$(realpath --relative-to=. $DATA_ROOT)
# # target data path
# target_data_path="./refactored_dataset_for_flevel"
# # puts the wav files into a structured folder, separating classes, as expected by pyAudioAnalysis library.
# ./gather_final_dataset.sh $source_data_path $target_data_path/final/fu "fu"
# # copying models into the root dir
# cp ./models/flevel* .
# # run the inference using the pretained models
# python ./src/apply_existing_model.py -d $target_data_path/final/fu/test -m "flevel-randomforest-final" -c "fu" --predict_on_private
# # return bach to root
# cd ../../

# # filling_type
# cd ./filling_type/CORSMAL-pyAudioAnalysis/
# chmod +x ./gather_final_dataset.sh
# # retriveing the relative path to the dataset from this folder
# source_data_path=$(realpath --relative-to=. $DATA_ROOT)
# # target data path
# target_data_path="./refactored_dataset_for_ftype"
# # puts the wav files into a structured folder, separating classes, as expected by pyAudioAnalysis library.
# ./gather_final_dataset.sh $source_data_path $target_data_path/final/fi "fi"
# # copying models into the root dir
# cp ./models/ftype* .
# # run the inference using the pretained models
# python ./src/apply_existing_model.py -d $target_data_path/final/fi/test -m "ftype-randomforest-final" -c "fi" --predict_on_private
# # return bach to root
# cd ../../
