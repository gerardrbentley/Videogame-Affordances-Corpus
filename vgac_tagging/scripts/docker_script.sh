#!/bin/bash
echo 'DOCKER SCRIPT CALLED'
python manage.py ingest_files
echo 'DONE INGESTING, STARTING SERVER'
python manage.py runserver
