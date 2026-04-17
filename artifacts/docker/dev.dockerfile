FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

SHELL [ "/bin/bash", "--login", "-c" ]

# Set locale.
RUN apt-get update -y && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# Install tools for installers.

# install colmap directly inside the image
# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive
ARG COLMAP_VERSION=78f1eefacae542d753c2e4f6a26771a0d976227d
ARG CUDA_ARCHITECTURES="60;70;75;80;86"

# Prepare and empty machine for building
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y \
    git \
    ninja-build \
    sudo \
    wget \
    unzip \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libcgal-dev \
    gcc-10 \
    g++-10 \
    libflann-dev \
    libsqlite3-dev \
    libqt5opengl5-dev \
    python3-pip=20.0.2* \
    python3-tk \
    daemontools \
    libgl1-mesa-glx \
    libpng-dev libjpeg-dev libtiff-dev libxxf86vm1 libxxf86vm-dev libxi-dev libxrandr-dev coinor-libclp-dev \
    libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*


ENV CC=/usr/bin/gcc-10
ENV CXX=/usr/bin/g++-10
ENV CUDAHOSTCXX=/usr/bin/g++-10

# installl cmake
COPY installers/cmake.sh /tmp/installers/
RUN bash /tmp/installers/cmake.sh && rm /tmp/installers/cmake.sh

# installl eigen ceres
COPY installers/ceres.sh /tmp/installers/
RUN bash /tmp/installers/ceres.sh && rm /tmp/installers/ceres.sh

# install colmap
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git reset --hard b0f3c6d0f6550bd1f40942a119feb7fd2e96ff5e && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DFETCH_POSELIB=OFF && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap

# install glomap
COPY installers/glomap.sh /tmp/installers/
RUN bash /tmp/installers/glomap.sh && rm /tmp/installers/glomap.sh

# installl openMVG
# COPY installers/openMBG.sh /tmp/installers/
# RUN bash /tmp/installers/openMBG.sh && rm /tmp/installers/openMBG.sh

# installl insta360_sdk
COPY installers/insta360_sdk.sh /tmp/installers/
RUN bash /tmp/installers/insta360_sdk.sh && rm /tmp/installers/insta360_sdk.sh

# installl install_exiftools
COPY installers/install_exiftools.sh /tmp/installers/
RUN bash /tmp/installers/install_exiftools.sh && rm /tmp/installers/install_exiftools.sh

# install miniconda
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py312_24.5.0-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

COPY environment.yml /tmp/
RUN conda env create --file /tmp/environment.yml

RUN pip install jupyterlab

RUN conda activate gaussian_splatting
