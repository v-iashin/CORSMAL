# The CORSMAL Challenge (on Intelligent Sensing Summer School 2020)

The 🏆 Winning Solution (in all sub-tasks) to the 3-day competition on the CORSMAL challenge organized among participants of the [Intelligent Sensing Summer School (1–4 September 2020)](http://cis.eecs.qmul.ac.uk/school2020.html).

## Team 👋
- Gokhan Solak ([LinkedIn](https://www.linkedin.com/in/gkhnsolak/), g.solak@qmul.ac.uk)
- Francesca Palermo ([LinkedIn](https://www.linkedin.com/in/francesca-palermo-a9107a40/), f.palermo@qmul.ac.uk)
- Claudio Coppola ([LinkedIn](https://www.linkedin.com/in/clcoppola/) c.coppola@qmul.ac.uk)
- Vladimir Iashin ([LinkedIn](https://www.linkedin.com/in/vladimir-iashin/), vladimir.iashin@tuni.fi)

## Task
The CORSMAL challenge  focuses on the estimation  of the weight of containers which depends  on the presence  of a filling and  its  amount  and type,  in addition  to the  container  capacity.  Participants  should determine  the  physical  properties  of  a  container  while  it  is  manipulated  by  a  human, when both containers  and fillings  are not known a priori.

Technically, the main task requires to estimate the  overall  filling  mass  estimation. This quantity can be estimated by solving three sub-tasks:
- Container capacity estimation (any positive number)
- Filling type classification (boxes: pasta, rice; glasses/cups: water, pasta, rice; or nothing (empty))
- Filling level classification (0, 50, 90%)

## [CORSMAL dataset](http://corsmal.eecs.qmul.ac.uk/containers_manip.html) 🥤🍚🥛

- The dataset consists of **15 containers**: 5 drinking cups, glasses, and food boxes. These containers are made of different materials, such as **plastic**, **glass**, and **paper**.
- A container can be **filled with water** (only glasses and cups), **rice or pasta** at 3 different levels of **0, 50, and 90%** with respect to the capacity of the container.
- All different combinations of containers are executed by a **different subject** (12) for each **background** (2) and **illumination** condition (2). The total number of configurations is **1140**.
- Each event in the dataset is acquired with several sensors, making the CORSMAL dataset to be multi-modal
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
1. Clone the repo recursively (with all submodule): `git clone --recursive https://github.com/v-iashin/CORSMAL.git`.
2. Run `./main.py` (requires `pandas`, though) it will take the predictions from each subtask folder and form the final submission file.

To train and obtain the individual prediction files, follow the instructions provided in `README` in individual sub-tasks folders.
Each folder has its environment, so you will need to install each of them and run the training/prediction procedure. Please note, it might give you _slightly_ different results depending on your hardware. We observed that the change is 🤏 `<0.01`.
