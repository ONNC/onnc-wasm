FROM onnc/onnc-wasm-backend-community:base

RUN sudo sed -i 's/archive.ubuntu.com/ftp.ubuntu-tw.net/' /etc/apt/sources.list && \
    sudo apt-get update && \
    sudo apt-get install -y python3 python3-pip && \
    sudo rm -rf /var/lib/apt/lists/* && \
    sudo apt-get clean

RUN pip3 install onnxruntime numpy

RUN mkdir -p /home/onnc/models/

COPY ./motd.txt /etc/motd

RUN echo "cat /etc/motd" >> ~/.bashrc

USER onnc

WORKDIR /home/onnc/workspace

ENV PATH="/opt/wasi-sdk/bin:${PATH}"