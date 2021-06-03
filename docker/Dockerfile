FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu18.04

RUN apt-get update \
 && apt-get install --no-install-recommends -y locales \
        python3-dev python3-pip python3-setuptools python3-wheel \
 && locale-gen en_US.UTF-8 \
 && python3 -m pip install -U pip

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
