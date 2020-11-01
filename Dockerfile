FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update
RUN apt-get install -y sudo=1.8.21p2-3ubuntu1.3

RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu

RUN sudo apt-get update
RUN sudo apt-get -qq install \
  curl=7.58.0-2ubuntu3.10 \
  vim=2:8.0.1453-1ubuntu1.4 \
  nano=2.9.3-2 \
  git=1:2.17.1-1ubuntu0.7 \
  zip=3.0-11build1 \
  libsndfile1=1.0.28-4ubuntu0.18.04.1 \
  libglib2.0-0=2.56.4-0ubuntu0.18.04.6 \
  libsm6=2:1.2.2-1 \
  libxext6=2:1.3.3-1 \
  libxrender-dev=1:0.9.10-1

WORKDIR /home/ubuntu/

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh
RUN bash ./Miniconda3-4.7.12-Linux-x86_64.sh -b
ENV PATH="/home/ubuntu/miniconda3/bin:$PATH"
RUN echo ". /home/ubuntu/miniconda3/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash
RUN conda config --set auto_activate_base false

RUN git clone --recursive https://github.com/v-iashin/CORSMAL.git

WORKDIR /home/ubuntu/CORSMAL/

# TODO
RUN git checkout docker

RUN curl https://storage.googleapis.com/audioset/vggish_model.ckpt -o ./filling_level/vggish/video_features/models/vggish/checkpoints/vggish_model.ckpt
RUN conda env create -f ./filling_level/vggish/video_features/conda_env_vggish.yml
RUN conda env create -f ./filling_level/r21d_rgb/video_features/conda_env_r21d.yml
RUN conda env create -f ./filling_level/vggish/environment.yml
RUN conda env create -f ./filling_level/CORSMAL-pyAudioAnalysis/environment.yml
RUN conda env create -f ./capacity/LoDE_linux.yml

RUN conda clean -afy

RUN cd ./filling_level/CORSMAL-pyAudioAnalysis && curl -O https://raw.githubusercontent.com/v-iashin/CORSMAL-pyAudioAnalysis/master/gather_final_dataset.sh
RUN cd ./filling_type/CORSMAL-pyAudioAnalysis && curl -O https://raw.githubusercontent.com/v-iashin/CORSMAL-pyAudioAnalysis/master/gather_final_dataset.sh
RUN cd ./filling_level/CORSMAL-pyAudioAnalysis/src && curl -O https://raw.githubusercontent.com/v-iashin/CORSMAL-pyAudioAnalysis/master/src/apply_existing_model.py
RUN cd ./filling_type/CORSMAL-pyAudioAnalysis/src && curl -O https://raw.githubusercontent.com/v-iashin/CORSMAL-pyAudioAnalysis/master/src/apply_existing_model.py
