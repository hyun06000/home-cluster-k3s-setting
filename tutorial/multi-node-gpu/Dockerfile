FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

RUN HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_CPU_OPERATIONS=MPI \
    HOROVOD_WITH_MPI=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    pip3 install horovod

COPY ./train_script.py /app/train_script.py

WORKDIR /app
