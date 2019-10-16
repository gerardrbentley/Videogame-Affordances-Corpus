#!/bin/bash
python manage.py ingest_files
echo 'DONE INGESTING'
python manage.py runserver
