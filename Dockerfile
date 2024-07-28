FROM pytorch/pytorch:latest

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y ffmpeg build-essential

RUN mkdir /app
WORKDIR /app

COPY ./requirements.txt .
RUN python -m pip install -U -r /app/requirements.txt --no-cache-dir

# Dev
RUN python -m pip install -U \
    pytest \
    black \
    ipdb