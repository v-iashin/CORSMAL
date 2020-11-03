## pyAudioAnalysis

Installation and run instructions can be found in `./CORSMAL-pyAudioAnalysis/README.md`. Run the lines corresponding to `fu` (filling level).

Result: accuracy (avg accuracy across folds): `0.70`

## VGGish

1. (optional since we have them in `./vggish_features` already) Run `bash ./extract_features.sh` to extract Vggish features using `./video_features` (follow the guidelines in the `README.md` to install the env and download the vggish model checkpoint). Note, when you will extract the features the hash of some features can be different to the ones in the repo due to numerical errors (average difference is < 1e-6).
2. If you haven't yet, install the environment in `environment.yml` (`conda env create -f environment.yml`). If you are not planning to use GPU see a note below.
3. Run `main.ipynb` or `main.py` (use `use_pretrained = True` if you don't want to re-train the models). Predictions are saved into `./predictions` folder for each fold and phase (train/valid/test). Probs for each class is also provided.

After installing the environment we will need to change the `pytorch-gpu` on `pytorch-cpu`: `conda activate corsmal && conda uninstall pytorch torchvision && conda install pytorch torchvision cpuonly -c pytorch` and replace the device in `./predictions/cfg.txt` to `cpu`.

If you decided to re-train the model, switch `use_pretrained` to `False`. The expected F1 at the end of training: `0.755171` (results migh vary after 3rd digit).

## R(2+1)D

The procedure is quite similar to VGGish. Extract `R(2+1)D RGB` instead of `VGGish` features and the rest is the same. By the way, you don't need to install new environment as both `environment.yml` are identical.

Result: F1 (avg F1 across folds): `0.747354` (results migh vary after 3rd digit)

## Evaluation
For filling type and level classification, we employ 3-Fold validation, preserving the box/glass/cup ratio as in the test set.
