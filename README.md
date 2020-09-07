# The CORSMAL Challenge (on Intelligent Sensing Summer School 2020)

This is the code for the challenging 3 days competition among participants of the [Intelligent Sensing Summer School (1–4 September 2020)](http://cis.eecs.qmul.ac.uk/school2020.html).

The CORSMAL challenge  focuses on the estimation  of the weight of containers which depends  on the presence  of a filling and  its  amount  and type,  in addition  to the  container  capacity.  Participants  will determine  the  physical  properties  of  a  container  while  it  is  manipulated  by  a  human, when both containers  and fillings  are not known a priori.

## Team
- Gokhan Solak ([LinkedIn](https://www.linkedin.com/in/gkhnsolak/), g.solak@qmul.ac.uk)
- Francesca Palermo ([LinkedIn](https://www.linkedin.com/in/francesca-palermo-a9107a40/), f.palermo@qmul.ac.uk)
- Claudio Coppola (c.coppola@qmul.ac.uk)
- Vladimir Iashin ([LinkedIn](https://www.linkedin.com/in/vladimir-iashin/), vladimir.iashin@tuni.fi)

## Task
The main task is to estimate the  overall  filling  mass  estimation. This quantity can be estimated by solving three sub-tasks:
- Container capacity estimation (any positive number)
- Filling type classification (boxes: pasta, rice; glasses/cups: water, pasta, rice)
- Filling level classification (0, 50, 90%)

## Dataset
[CORSMAL dataset](http://corsmal.eecs.qmul.ac.uk/containers_manip.html) (Audio, RGB, IMU, Depth, IR)

It has 12 containers (4 glasses, cups, and boxes; 9 train + 3 test).

## Evaluation
We used 3-Fold validation where possible, preserving the box/glass/cup ratio as in the test set.

## Organization or this repo

Each folder in this repo correspond to a dedicated task. Filling level and types have two or three different approaches and, hence, each approach has an individual folder.

- `/capacity`
- `/filling_level`
    - `/r21d_rgb`
    - `/vggish`
    - `README.md`
- `/filling_type`
    - `/CORSMAL-pyAudioAnalysis`
    - `/r21d_rgb`
    - `/vggish`
    - `README.md`

## How to run
1. Clone the repo: `git clone --recursive https://github.com/v-iashin/CORSMAL.git`. Mind the `--recursive` flag (if you forgot the flag in the first place, just run `git submodule update --init`);
2. Run `./main.py` (requires `pandas` though) it will take the predictions from each subtask folder and form the final submission file.

Please follow the instructions provided in individual sub-tasks folders to train and obtain the preds. Each folder has its own environment, you will need to install those.
