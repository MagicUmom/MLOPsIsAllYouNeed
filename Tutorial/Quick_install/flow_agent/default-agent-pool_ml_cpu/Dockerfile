FROM prefecthq/prefect:2.10.9-python3.10

WORKDIR /root/flows

COPY requirements_sys.txt .
RUN pip3 install -r requirements_sys.txt

COPY requirements.txt .
RUN pip3 install -r requirements.txt
