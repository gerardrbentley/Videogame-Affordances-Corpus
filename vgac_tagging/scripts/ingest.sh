#!/usr/bin/env bash

docker-compose run -v ../games:/games app python manage.py ingestall /games
