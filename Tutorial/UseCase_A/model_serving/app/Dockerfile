FROM python:3.9.10-buster

WORKDIR /app

COPY . /app

ENV FLASK_APP=app.py

RUN apt-get update && apt-get -y install ffmpeg libsm6 libxext6 && \
    pip install -r /app/requirements_app.txt