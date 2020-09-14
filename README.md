# The CORSMAL Challenge (on Intelligent Sensing Summer School 2020)

The 🏆 Winning Solution (in all sub-tasks) to the 3-day competition on the CORSMAL challenge organized among participants of the [Intelligent Sensing Summer School (1–4 September 2020)](http://cis.eecs.qmul.ac.uk/school2020.html).

## Team 👋
- Gokhan Solak ([LinkedIn](https://www.linkedin.com/in/gkhnsolak/), g.solak@qmul.ac.uk)
- Francesca Palermo ([LinkedIn](https://www.linkedin.com/in/francesca-palermo-a9107a40/), f.palermo@qmul.ac.uk)
- Claudio Coppola ([LinkedIn](https://www.linkedin.com/in/clcoppola/), c.coppola@qmul.ac.uk)
- Vladimir Iashin ([LinkedIn](https://www.linkedin.com/in/vladimir-iashin/), vladimir.iashin@tuni.fi)

## Task
The CORSMAL challenge  focuses on the estimation  of the weight of containers which depends  on the presence  of a filling and  its  amount  and type,  in addition  to the  container  capacity.  Participants  should determine  the  physical  properties  of  a  container  while  it  is  manipulated  by  a  human, when both containers  and fillings  are not known a priori.

Technically, the main task requires to estimate the  overall  filling  mass  estimation. This quantity can be estimated by solving three sub-tasks:
- Container capacity estimation (any positive number)
- Filling type classification (boxes: pasta, rice; glasses/cups: water, pasta, rice; or nothing (empty))
- Filling level classification (0, 50, 90%)

## [CORSMAL dataset](http://corsmal.eecs.qmul.ac.uk/containers_manip.html) 🥤📘🥛
- The dataset consists of **15 containers**: 5 drinking cups, glasses, and food boxes. These containers are made of different materials, such as **plastic**, **glass**, and **paper**.
- A container can be **filled with water** (only glasses and cups), **rice or pasta** at 3 different levels of **0, 50, and 90%** with respect to the capacity of the container.
- All different combinations of containers are executed by a **different subject** (12) for each **background** (2) and **illumination** condition (2). The total number of configurations is **1140**.
- Each event in the dataset is acquired with several sensors, making the CORSMAL dataset to be **multi-modal**
    - 4 cameras (1280x720@30Hz):
        - RGB,
        - narrow-baseline stereo infrared,
        - depth images,
        - inertial measurements (IMU)
        - calibration
    - 8-channel 44.1 kHz audio;

## Evaluation
For filling type and level classification, we employ 3-Fold validation, preserving the box/glass/cup ratio as in the test set.

## Folder organization

Each folder in this repo corresponds to a dedicated task. Filling level and types have two or three different approaches and, hence, each approach has an individual folder.
- `/capacity`
- `/filling_level`
    - `/CORSMAL-pyAudioAnalysis`
    - `/r21d_rgb`
    - `/vggish`
    - `README.md`
- `/filling_type`
    - `/CORSMAL-pyAudioAnalysis`
    - `/r21d_rgb`
    - `/vggish`
    - `README.md`

## How to run

Clone the repo recursively (with all submodule):
```bash
git clone --recursive https://github.com/v-iashin/CORSMAL.git
```
All python environements can be installed via `conda` and tested, at least, on Linux. [How to install conda follow this guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) – `miniconda3` is recommended. Once `conda` is installed, run the following commands to install all required environments for each sub-task
```bash
# it will create `LoDE` environment
conda env create -f ./capacity/LoDE_linux.yml
# it will create `corsmal` environment
conda env create -f ./filling_level/vggish/environment.yml
# it will create `pyAudioAnalysis` environment
conda env create -f ./filling_level/CORSMAL-pyAudioAnalysis/environment.yml
```
Installation should not give you any error. If some package fails to be installed, please let us know in Issues.

How to use the installed environments? One way is to activate the installed environments and run the scripts from `python` command.
Another way is to use the path to `python` in a specfic environment to run your scripts:
```bash
$ conda activate LoDE
(LoDE) $ python  YOUR_PYTHON_SCRIPT.py
# which is equivalant to
$ ~/miniconda3/envs/LoDE/bin/python  YOUR_PYTHON_SCRIPT.py
```

To reprodce the results please follow the guidelines provided in `README`s:
1. Capacity: `./capacity/README.md`
2. Filling Level: `./filling_level/CORSMAL-pyAudioAnalysis/README.md` (lines with `fu`) and `./filling_level/README.md`
3. Filling Type: `./filling_type/CORSMAL-pyAudioAnalysis/README.md` (lines with `fi`) and `./filling_type/README.md`
4. Finally, run `./main.py` (using, i.e. `corsmal` conda env) it will take the predictions from each subtask folder and form the final submission file

Please note, we undertook extreme care to make sure our results are reproducible by fixing the seeds, sharing the pre-trained models, and package versions. However, the training on another hardware might give you _slightly_ different results. We observed that the change is 🤏 `<0.01`.
