FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# setting python 
WORKDIR /app
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y software-properties-common && \
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN curl -sSL https://install.python-poetry.org | python3.10 - --preview
RUN pip3 install --upgrade requests
RUN ln -fs /usr/bin/python3.10 /usr/bin/python

# install python packages
RUN apt-get update && \
    apt-get install -y python3.10-distutils

# download pytorch
RUN pip3 install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# download packages
WORKDIR /root/flows

COPY requirements_sys.txt .
RUN pip3 install -r requirements_sys.txt

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN apt-get install -y git