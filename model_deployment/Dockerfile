FROM python:slim-buster

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBUG 0

RUN apt update && \
    apt upgrade -y
RUN apt install ffmpeg libsm6 libxext6  -y
RUN pip3 install torch==1.8.1+cpu \
    torchvision==0.9.1+cpu \
    torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY ./DockerEssential.txt .
RUN pip3 install -r DockerEssential.txt

COPY . .
RUN python3 manage.py migrate
CMD python3 manage.py runserver 0.0.0.0:8000