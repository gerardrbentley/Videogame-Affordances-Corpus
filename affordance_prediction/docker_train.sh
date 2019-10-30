#!/bin/bash
echo "BEGIN TRAIN SCRIPT"
rclone config file
rclone config dump

python newtrain.py $@
