#!/bin/bash

# Expects 1 arg, an absolute path to Video Directory containing recording.mkv in current working directory
# Makes $PWD/per_game_screenshots/$SANITIZED_GAME_NAME and adds all screenshots to that folder
# Suggested Usage: $ ls -d $PWD/videos/* | parallel bash mkv_to_pngs.sh {}
# add any word after {} to trigger copy of mkv recording to per_game_videos folder
# TODO option to UUID in this step  
if [ "$1" != "" ]; then
    DIR=`pwd`
    echo "$DIR"
    echo "slice .mkv file in trace folder $1 into per_game_screenshots"
    FOLDER=$(basename "$1")
    # echo "Original folder: $FOLDER"

    # remove everything after _ (trace tag)
    CLEAN_GAME="${FOLDER%%_*}"
    # sub _ for spaces
    CLEAN_GAME=${CLEAN_GAME// /_}
    # Remove non a-z, A-Z, 0-9, or _
    CLEAN_GAME=${CLEAN_GAME//[^a-zA-Z0-9_]/}
    # Replace double __ with _ (from names like pokemon - blue)
    CLEAN_GAME=${CLEAN_GAME//__/_}
    # lowercase all
    CLEAN_GAME=${CLEAN_GAME,,}
    echo "Old folder: $FOLDER | New folder: $CLEAN_GAME"

    mkdir -p "$DIR/per_game_screenshots/$CLEAN_GAME"
    
    # Use $1 for path $FOLDER to avoid screenshot naming collisions
    ffmpeg -i "$1"/recording.mkv -r 2 "$DIR"/per_game_screenshots/"$CLEAN_GAME"/sc-"$FOLDER"-%04d.png
    
    if [ "$2" != "" ]; then
        mkdir -p "$DIR/per_game_videos/$CLEAN_GAME"
        NEXT=`ls -f "$DIR/per_game_videos/$CLEAN_GAME" | wc -l`
        # echo "$NEXT"
        cp "$1"/recording.mkv "$DIR/per_game_videos/$CLEAN_GAME/$NEXT.mkv"
    fi
    echo --------------------------------
else
    echo "No param 1"
fi
