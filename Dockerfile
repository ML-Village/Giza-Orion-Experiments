FROM python:3.11.5

WORKDIR /home

RUN pip install --upgrade pip \
    giza-cli
