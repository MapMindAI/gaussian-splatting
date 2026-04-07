FROM determinedai/environments:cuda-11.3-pytorch-1.12-tf-2.11-gpu-mpi-0.24.0

# Set locale.
RUN apt-get update -y && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# Install tools for installers.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential=12.8* \
    cmake=3.16.3* \
    curl=7.68.0* \
    git=1:2.25.1* \
    git-lfs \
    pkg-config=0.29.1* \
    software-properties-common=0.99.9* \
    unzip=6.0* \
    wget=1.20.3* \
    libcurl4-openssl-dev \
    sshfs \
    sudo \
    vim \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# install libraries
# includes https://github.com/facebookresearch/dinov2/blob/main/requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  opencv-contrib-python==4.8.0.76 \
  scikit-learn==0.24.2 \
  accelerate==0.12.0 \
  pose-transform==0.3.1 \
  setuptools==59.5.0 \
  joblib \
  plyfile \
  tqdm \
  mmsegmentation==0.27.0 \
  ultralytics \
  pyzbar \
  omegaconf

RUN pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
