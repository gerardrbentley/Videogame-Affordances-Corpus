#!/bin/bash
for subdir in *
do
  for i in $subdir/img/*.png
  do uuid=$(uuidgen -r) && mv -- "$i" "$subdir/img/$uuid.png" \
    && file=$(basename $i .png) && mv -- "$subdir/label/$file.npy" "$subdir/label/$uuid.npy"
  done
  for j in $subdir/tile_img/*.png
  do uuid=$(uuidgen -r) && mv -- "$j" "$subdir/tile_img/$uuid.png"
  done
done
