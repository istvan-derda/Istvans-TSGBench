FROM docker.io/python:3.7-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends rsync && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir tensorflow-cpu==1.15
RUN pip install --no-cache-dir numpy==1.21.5
RUN pip install --no-cache-dir pandas==1.3.5
RUN pip install --no-cache-dir scipy==1.6.1
RUN pip install --no-cache-dir statsmodels==0.13.5
RUN pip install --no-cache-dir matplotlib==3.5.3
RUN pip install --no-cache-dir scikit-learn==1.0.2
RUN pip install --no-cache-dir seaborn==0.12.2
RUN pip install --no-cache-dir pyprojroot==0.3.0
RUN pip install --no-cache-dir ipykernel
RUN pip install --no-cache-dir protobuf==3.20.0
RUN pip install --no-cache-dir torch==1.13.1
RUN pip install --no-cache-dir dtaidistance==2.3.12
RUN pip install --no-cache-dir plotly==5.18
RUN pip install --no-cache-dir jinja2==3.1
RUN pip install --no-cache-dir nbformat==5.8
RUN pip install --no-cache-dir scikit-posthocs
RUN pip install --no-cache-dir kaleido

RUN apt-get update && \
    apt-get install -y --no-install-recommends vim && \
    rm -rf /var/lib/apt/lists/*
