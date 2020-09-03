## VGGish

1. Extract Vggish features using `./video_features` (follow the guidelines in the folder to install the env)
    1. For that you will need to construct file paths. see `form_filenames.py`
2. Install the environment in `environment.yml`
3. Run `main.py` or `main.ipynb` (adjust the default agrs in the file or use cmd line args).
4. Predictions are saved into `./predictions` folder for each fold, phase (train/valid/test). Props for each class is also provided.

Result: F1 (avg F1 across folds): 0.755
