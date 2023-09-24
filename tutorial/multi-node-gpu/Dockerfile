FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    python3-dev \
    python3-setuptools \
    python3-pip \
    libffi-dev \
    python-cffi \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && \
    wget https://github.com/Kitware/CMake/releases/download/v3.13.0/cmake-3.13.0.tar.gz && \
    tar -xzvf cmake-3.13.0.tar.gz && \
    cd cmake-3.13.0 && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install

RUN pip3 install mpi4py

RUN ln -s $(which pip3) /usr/bin/pip

RUN HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_CPU_OPERATIONS=MPI \
    HOROVOD_WITH_MPI=1 \
    HOROVOD_NCCL_LINK=SHARED \
    HOROVOD_GPU_OPERATIONS=NCCL \
    HOROVOD_NCCL_INCLUDE=/usr/include \
    HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu \
    pip install horovod

COPY ./train_script.py /app/train_script.py

WORKDIR /app

CMD ["python3", "train_script.py"]
