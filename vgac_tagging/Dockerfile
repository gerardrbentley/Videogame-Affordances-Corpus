FROM python:3.7
LABEL maintainer "FAIM Lab: Omar Zintan <ozma2018@mymail.pomona.edu>"

COPY ./requirements.txt /
RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app

ENV FLASK_ENV=docker
EXPOSE 80
