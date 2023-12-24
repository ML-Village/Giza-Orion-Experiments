FROM python:3.11.5

WORKDIR /home

RUN pip install --upgrade pip \
    giza-cli \
    onnx==1.14.1 \
    torch==2.1.0 \
    torchvision==0.16.0
