FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

RUN apt-get update && apt-get install libgl1 -y
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install libpq-dev gcc -y
RUN apt-get install python3 python3-dev python3-pip -y

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install tzdata -y
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata

RUN apt-get install libglib2.0 libglib2.0-dev -y

RUN apt-get install libsm6 libxext6 -y

COPY . /app

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN pip3 install -r requirements_gpu.txt

WORKDIR /app/yolov4-main

RUN make clean && make

WORKDIR /app


