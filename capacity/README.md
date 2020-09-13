# Multi-view shape estimation of transparent containers

## Description
Extension of the LoDE (Localisation and Object Dimensions Estimator).
LoDE is a method for jointly localising container-like objects and estimating their dimensions using
two wide-baseline, calibrated RGB cameras. Under the assumption of circular
symmetry along the vertical axis, LoDE estimates the dimensions of an object
with a generative 3D sampling model of sparse circumferences, iterative shape
fitting and image re-projection to verify the sampling hypotheses in each camera
using semantic segmentation masks (Mask R-CNN).
In addition, the automatic frame extraction from the complete videos database has been included.

- [LoDE webpage](http://corsmal.eecs.qmul.ac.uk/LoDE.html)
- [CORSMAL Containers Manipulation Dataset](http://corsmal.eecs.qmul.ac.uk/containers_manip.html)

## Installation
Create the conda environment from file ([more info on how to install miniconda](https://docs.conda.io/en/latest/miniconda.html)). We tested our code on Windows 10 and Ubuntu 18.06

```bash
# if you are on windows use
conda env create --file LoDE_windows.yml
# if you are on linux use
conda env create --file LoDE_linux.yml
source activate LoDE
```

## Preparing the CORSMAL Containers Manipulation dataset
Download the CORSMAL Containers Manipulation dataset (http://corsmal.eecs.qmul.ac.uk/containers_manip.html)
An example has been already downloaded and inserted in the Github repository.

The video dataset should be in the same working directory than LoDE.
We expect your __video_database__ folder to be organized like this. So you will only need to unzip `depth`, `ir`, and `rgb` from the original dataset.
```
./video_database
├── 10
│   ├── depth
│   ├── ir
│   └── rgb
├── 11
│   ├── depth
│   ├── ir
│   └── rgb
└── 12
    ├── depth
    ├── ir
    └── rgb
```

The code will extract frames from the videos presents in __video_database__ and will move them into the corresponding __dataset__ folder. For example, using this script
```bash
#!/bin/bash
CORSMAL_PATH="/YOUR/PATH/WITH/NO/TRAILING/SLASH"
for container_id in {1..12}
do
cp -r $CORSMAL_PATH"/"$container_id "./capacity/video_datatabase/"
done
```

# Run on the Test Set
Run LoDE on the whole testing set
```bash
python main.py
```

## Output
This version of LoDE will outputs four results:
* Dimensions estimation of the height of the container in milimeters in results/estimation.txt
* Dimensions estimation of the width of the container in milimeters in results/estimation.txt
* Dimensions estimation of the container capacity of the container in milliliters in results/estimation.txt
* Visual representation of the container shape in results/*.png. The visual representation can be removed by omitting the --draw commands


## Citation
If you use this data, please cite:
A. Xompero, R. Sanchez-Matilla, A. Modas, P. Frossard, and A. Cavallaro,
_Multi-view shape estimation of transparent containers_, Published in the IEEE
2020 International Conference on Acoustics, Speech, and Signal Processing,
Barcelona, Spain, 4-8 May 2020.

Bibtex:
@InProceedings{Xompero2020ICASSP,
  TITLE   = {Multi-view shape estimation of transparent containers},
  AUTHOR  = {A. Xompero, R. Sanchez-Matilla, A. Modas, P. Frossard, and A. Cavallaro},
  BOOKTITLE = {IEEE 2020 International Conference on Acoustics, Speech, and Signal Processing},
  ADDRESS	       = {Barcelona, Spain},
  MONTH		       = "4--8~" # MAY,
  YEAR		       = 2020
}


## Licence
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
