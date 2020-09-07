## VGGish

1. Run `bash ./extract.sh` to extract Vggish features using `./video_features` (follow the guidelines in the `README.md` to install the env and download the vggish model checkpoint). Alternativelly, use `./vggish_features` folder. Note, when you will extract the features the hash of some features can be different to the ones in the repo due to numerical errors (average difference is < 1e-6).
2. Install the environment in `environment.yml` (`conda env create -f environment.yml`)
3. Run `main.ipynb` or `main.py` (compare the output with the cell's comments, F1: 0.755171). Predictions are saved into `./predictions` folder for each fold and phase (train/valid/test). Props for each class is also provided.

## R(2+1)D

The procedure is quite similar to VGGish. Extract `R(2+1)D RGB` instead of `VGGish` features and the rest is the same. By the way, you don't need to install new environment as both `environment.yml` are identical.

Result: F1 (avg F1 across folds): 0.747354
