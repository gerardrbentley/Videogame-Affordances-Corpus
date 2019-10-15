#!/bin/bash

python manage.py recreate_db
python manage.py ingest_files
python manage.py runserver
