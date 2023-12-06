FROM prefecthq/prefect:2.11.0-python3.10

WORKDIR /root/flows

COPY ./flow_scheduler/requirements_sys.txt .
RUN pip3 install -r requirements_sys.txt

ARG FLOW_DIR

COPY ${FLOW_DIR}/requirements.txt .
RUN pip3 install -r requirements.txt

COPY ${FLOW_DIR} .

COPY ./flow_scheduler/setup.py .