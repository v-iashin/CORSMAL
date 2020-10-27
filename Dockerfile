FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update
RUN apt-get install -y sudo

RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu

RUN sudo apt-get update
RUN sudo apt-get -qq install curl git zip libsndfile1

WORKDIR /home/ubuntu/

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh
RUN bash ./Miniconda3-4.7.12-Linux-x86_64.sh -b
ENV PATH="/home/ubuntu/miniconda3/bin:$PATH"
RUN echo ". /home/ubuntu/miniconda3/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash
RUN conda config --set auto_activate_base false

RUN git clone --recursive https://github.com/v-iashin/CORSMAL.git

WORKDIR /home/ubuntu/CORSMAL/
RUN curl https://storage.googleapis.com/audioset/vggish_model.ckpt -o ./filling_level/vggish/video_features/models/vggish/checkpoints/vggish_model.ckpt
RUN conda env create -f ./filling_level/vggish/video_features/conda_env_vggish.yml
RUN conda env create -f ./filling_level/r21d_rgb/video_features/conda_env_r21d.yml
RUN conda env create -f ./filling_level/vggish/environment.yml
RUN conda env create -f ./filling_level/CORSMAL-pyAudioAnalysis/environment.yml
RUN conda env create -f ./capacity/LoDE_linux.yml

RUN conda clean -afy

RUN conda install -y vim -c conda-forge
