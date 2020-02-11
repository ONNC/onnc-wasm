FROM ubuntu:bionic

RUN sed -i 's/archive.ubuntu.com/ftp.ubuntu-tw.net/' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y sudo git wget curl unzip cmake build-essential libgoogle-glog-dev libprotoc-dev libboost-dev libjpeg-dev libboost-system-dev libboost-filesystem-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN groupadd --gid 1000 onnc \
    && useradd --uid 1000 --gid onnc --shell /bin/bash --create-home onnc --home-dir /home/onnc/ \
    && mkdir -p /etc/sudoers.d \
    && echo 'onnc ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers.d/onnc \
    && chmod 440 /etc/sudoers.d/onnc

# wasi-sdk
RUN wget https://github.com/CraneStation/wasi-sdk/releases/download/wasi-sdk-7/wasi-sdk_7.0_amd64.deb && dpkg -i wasi-sdk_7.0_amd64.deb && rm wasi-sdk_7.0_amd64.deb

USER onnc

# Copy & build

WORKDIR /home/onnc

ENV PATH="/opt/wasi-sdk/bin:${PATH}"
