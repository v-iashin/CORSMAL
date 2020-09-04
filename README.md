# The CORSMAL Challenge (on Intelligent Sensing Summer School 2020)

This is the code for the challenging 3 days competition among participants of the [Intelligent Sensing Summer School (1–4 September 2020)](http://cis.eecs.qmul.ac.uk/school2020.html).

The CORSMAL challenge  focuses on the estimation  of the weight of containers which depends  on the presence  of a filling and  its  amount  and type,  in addition  to the  container  capacity.  Participants  will determine  the  physical  properties  of  a  container  while  it  is  manipulated  by  a  human, when both containers  and fillings  are not known a priori.

## Team
- Gokhan Solak (g.solak@qmul.ac.uk)
- Francesca Palermo (f.palermo@qmul.ac.uk)
- Claudio Coppola (c.coppola@qmul.ac.uk)
- Vladimir Iashin (vladimir.iashin@tuni.fi)

## Task
The main task is to estimate the  overall  filling  mass  estimation. This quantity can be estimated by solving three sub-tasks:
- Container capacity estimation (any positive number)
- Filling type classification (boxes: pasta, rice; glasses/cups: water, pasta, rice)
- Filling level classification (0, 50, 90%)

## Dataset
[CORSMAL dataset](http://corsmal.eecs.qmul.ac.uk/containers_manip.html) (Audio, RGB, IMU, Depth, IR)

It has 12 containers (4 glasses, cups, and boxes; 9 train + 3 test).

## Evaluation
We used 3-Fold validation where possible, preserving the box/glass/cup ratio as in test set.

## Organization or this repo

Each folder in this repo correspond to a dedicated task. Filling level and types have two or three different approaches and, hence, each approach has an individual folder.

- `/capacity`
- `/filling_level`
    - `/r21d_rgb`
    - `/vggish`
    - `README.md`
- `/filling_type`
    - `/CORSMAL-audio-only-filling-type-analysis`
    - `/r21d_rgb`
    - `/vggish`
    - `README.md`

## How to run
Please follow the instructions provided in individual sub-tasks folders.
