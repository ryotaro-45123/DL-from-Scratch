FROM python:3.8.7
USER root
WORKDIR /workspace
RUN pip install update pip \
    pip install numpy==1.19.4 \
    pip install matplotlib==3.3.3