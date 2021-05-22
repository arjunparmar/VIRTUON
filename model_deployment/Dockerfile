FROM python:3.8-alpine
FROM devashishupadhyay/scikit-learn-docker:latest

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBUG 0

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install torch==1.8.1+cpu \
    torchvision==0.9.1+cpu \
    torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

RUN python3 manage.py collectstatic --noinput

CMD gunicorn server.wsgi:application --bind 0.0.0.0:$PORT