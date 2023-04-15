FROM python:3.7.16
# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /repo/
COPY . /repo
# RUN apt-get update
# RUN apt-get upgrade -y
# RUN apt install libosmesa6-dev libgl1-mesa-glx libglfw3 -y
RUN bash setup.sh
RUN pip install -r requirements.txt
# RUN python awake_mujoco.py
# RUN python docker_keep.py