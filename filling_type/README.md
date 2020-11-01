## pyAudioAnalysis

Installation and run instructions can be found in `./CORSMAL-pyAudioAnalysis/README.md`. Run the lines corresponding to `fi` (filling type).

Result: accuracy (avg accuracy across folds): `0.94`

## VGGish

Please follow the guideline in `../filling_level/README.md`. The code is very similar. Specifically, you just need to switch `task` and `output_dim` in `cfg`, this change is depicted in `./vggish/main.py`.

Result: F1 (avg F1 across folds): `0.912957` (results migh vary after 3rd digit)

## R(2+1)D

Please follow guideline in `../filling_level/README.md`. The code is very similar. Specifically, you just need to switch `task` and `output_dim` in `cfg`, this change is depicted in `./r21d_rgb/main.py`.

Result: F1 (avg F1 across folds): `0.673348` (results migh vary after 3rd digit)

## Evaluation
For filling type and level classification, we employ 3-Fold validation, preserving the box/glass/cup ratio as in the test set.
