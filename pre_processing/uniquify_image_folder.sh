#!/bin/bash

# for SUBDIR in ./per_game_screenshots_magic/*/
# do
#     FOLDER=$(basename "$SUBDIR")
#     echo "$FOLDER"

#     echo $CLEAN

#     counter=0
#     for IMG_FILE in "$SUBDIR"/*.png
#     do
#         counter=$((counter+1))
#     done
#     echo $counter
    
# done

# Expects 1 arg, an absolute path to a game_screenshots dir containing .pngs
# Removes duplicate images in folder 
# Suggested Usage: $ ls -d $PWD/per_game_screenshots/* | parallel bash uniquify_image_folder.sh {}
# TODO remove black / white / one color images
if [ "$1" != "" ]; then
    DIR=`pwd`
    echo "$DIR"
    echo "Uniquify image set in directory $1"
    FOLDER=$(basename "$1")
    # echo "Original folder: $FOLDER"

    bin0=0
    bin1=0
    bin2=0
    tmp="/home/gbkh2015/research/vgac_tagging/pre_processing/per_game_screenshots/harvest_moon_friends_of_mineral_town_usa/sc-Harvest Moon - Friends of Mineral Town (USA)_804a145c12e5_1573844824-0010.png"
    tmp2="/home/gbkh2015/research/vgac_tagging/pre_processing/per_game_screenshots/harvest_moon_friends_of_mineral_town_usa/sc-Harvest Moon - Friends of Mineral Town (USA)_804a145c12e5_1573844824-0005.png"
    test=`compare -metric RMSE "$tmp" "$tmp2" NULL: 2>&1 | tr -cs ".0-9" " " | cut -d\  -f2`
    echo "test: $test"
    for IMG_FILE in "$1"/*.png
    do
        # echo "$IMG_FILE"
        var=`compare -metric RMSE "$tmp" "$IMG_FILE" NULL: 2>&1 | tr -cs ".0-9" " " | cut -d\  -f2`
        # echo "$var"
        if (( $(echo "$var < 0.05" |bc -l) )); then
            bin0=$((bin0+1))
        elif (( $(echo "$var < 0.1" |bc -l) )); then
            bin1=$((bin1+1))
        elif (( $(echo "$var < 0.5" |bc -l) )); then
            bin2=$((bin2+1))
        fi
        # Test if files exist in double for loop
        # counter=$((counter+1))
    done
    echo "$bin0"
    echo "$bin1"
    echo "$bin2"
    # echo $counter
    # IMG_FILES="$1"/*.png
    echo --------------------------------
else
    echo "No param 1"
fi