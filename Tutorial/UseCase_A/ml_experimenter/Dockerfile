FROM jupyter/scipy-notebook:python-3.10

WORKDIR /root/setup
COPY requirements_sys.txt .
COPY requirements.txt .
RUN pip3 install -r requirements_sys.txt
RUN pip3 install -r requirements.txt