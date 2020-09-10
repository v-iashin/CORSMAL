## pyAudioAnalysis

Installation and run instructions can be found in `./CORSMAL-pyAudioAnalysis/README.md`. Run the lines corresponding to `fu` (filling level).

Result: accuracy (avg accuracy across folds): `0.70`

## VGGish

1. Run `bash ./extract_video_features.sh` to extract Vggish features using `./video_features` (follow the guidelines in the `README.md` to install the env and download the vggish model checkpoint). Alternativelly, use `./vggish_features` folder. Note, when you will extract the features the hash of some features can be different to the ones in the repo due to numerical errors (average difference is < 1e-6).
2. Install the environment in `environment.yml` (`conda env create -f environment.yml`). If you don't have a GPU (see a note below).
3. Run `main.ipynb` or `main.py` (compare the output with the cell's comments, F1: `0.755171`, results migh vary after 3rd digit). Predictions are saved into `./predictions` folder for each fold and phase (train/valid/test). Props for each class is also provided.

After installing the environment we will need to change the `pytorch-gpu` on `pytorch-cpu`: `conda activate corsmal && conda uninstall pytorch torchvision && conda install pytorch torchvision cpuonly -c pytorch` and replace the device in `./predictions/cfg.txt` to `cpu`.

## R(2+1)D

The procedure is quite similar to VGGish. Extract `R(2+1)D RGB` instead of `VGGish` features and the rest is the same. By the way, you don't need to install new environment as both `environment.yml` are identical.

Result: F1 (avg F1 across folds): `0.747354` (results migh vary after 3rd digit)
