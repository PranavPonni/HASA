FROM ros:noetic

ENV http_proxy http://www-proxy.waseda.jp:8080
ENV https_proxy http://www-proxy.waseda.jp:8080
ENV ftp_proxy http://www-proxy.waseda.jp:8080
ENV no_proxy "localhost,127.0.0.1,localaddress,.localdomain.com"

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt update && \
    apt upgrade -y && \
    apt install -y \
    wget \
    git \
    python3-pip \
    python3-tk \
    ffmpeg \
    glances \
    terminator \
    nautilus \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio torchsummary \
    pandas einops tensorboard ruamel.yaml tqdm imageio \
    wandb torchviz opencv-python matplotlib scikit-learn \
    "numpy<2"

RUN pip3 install pynput


WORKDIR /root/work
