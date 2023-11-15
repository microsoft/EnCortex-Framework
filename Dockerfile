FROM mcr.microsoft.com/azureml/curated/pytorch-1.10-ubuntu18.04-py38-cuda11-gpu

RUN mkdir /encortex
WORKDIR /encortex

RUN apt update && \
    apt install -y postgresql-client

COPY . .
RUN pip install .

#Temporary: TODO: remove this
ENV POSTGRES_URI 1
RUN pip install dfdb-0.0.1-py3-none-any.whl

RUN mkdir /workspace
WORKDIR /workspace