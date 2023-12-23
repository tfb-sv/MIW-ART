# hash:sha256:dee541f0c5b718695df040e79f30db827acce143fca597c3bd7685a984e7e060
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libxext6=2:1.3.4-0ubuntu1 \
        libxrender1=1:0.9.10-1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    ete3==3.1.3 \
    ipython==8.12.3 \
    matplotlib==3.7.4 \
    numpy==1.23.1 \
    pandas==2.0.3 \
    pubchempy==1.0.4 \
    rdkit==2022.3.5 \
    scikit-learn==1.3.2 \
    seaborn==0.13.0 \
    torch==2.1.2 \
    torchaudio==2.1.2 \
    torchvision==0.16.2 \
    tqdm==4.66.1
