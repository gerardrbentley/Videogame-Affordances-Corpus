# VGAC-Tagging

Current tagging tool for annotating images for VGAC

Link to Paper: [EXAG 2019](http://www.exag.org/papers/EXAG_2019_paper_13.pdf)

## Table of Contents

- [Docker Install](#docker)
- [Installation](#installation)
- [Pre-processing](#pre-processing)
- [Prediction](#prediction)
- [Background](#background)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)


## Docker Install
We use git lfs for large file storage of the dataset and pre-trained weights. Check if you have it installed with `git lfs install`. If this does not return 'Gif LFS Initialized' go download git lfs [here](https://git-lfs.github.com/)

The easiest way to get up and running is with Docker, these instructions will focus on that set up. Clone this repository's url

```
git clone GITHUBURL YOUR_FOLDER_NAME

cd YOUR_FOLDER_NAME/vgac_tagging
```

This folder should contain the docker-compose.yml file and `vgac_current.zip` (among everything else)

The most recent dataset is available in the zip file `vgac_current.zip`; unzip it so that the `games` directory is in the same level as the zip (double-click on mac, open in file manager and drag games folder on linux).

This contains game screenshots in .png format and affordance maps for legend of zelda and super mario 3 images in 9 channel .npy files.

Our current tagging process uses only tiles and screenshots. It requires pre-processing of tiles to be matched into images, currently only the super mario 3 tiles are complete and in pickled form. Tile and sprite ingestion from image files is possible, but not cleanly implemented.

To initialize the database and ingest all screenshots and tiles from the directory `games` (see directory structure below), run the following command (may or may not need sudo depending on your docker set up)

```
sudo docker-compose run --service-ports app ./scripts/wait-for-it.sh postgres:5432 -- ./scripts/docker_ingest.sh
```
(If this does not work you may need to fix docker or may need to make the .sh scripts executable. use `chmod +x scripts/*.sh` to do this for all of them)

This process should not take too long and should end with a 'DONE INGESTING' message in your terminal.

After that completes you can simply use the following to run the server on port 5000
```
sudo docker-compose up
```

If any unexpected behaviours occur, you can use the following to clean up the containers and persistent database volume
```
sudo docker-compose down --remove-orphans --volumes
```

expects the following directory structure
```
project_name
|   README.md
|___pre_processing
|
|___vgac_tagging
|   |   manage.py
|   |   requirements.txt
|   |   Dockerfile-dev
|   |   docker-compose.yml
|   |
|   |___scripts   
|   |   |   docker_ingest.sh
|   |   |   docker_script.sh
|   |
|   |___games
|       |___sm3
|       |   |   sm3_min_unique_lengths_offsets.csv
|       |   |   sm3_unique_set_NUMTILES.tiles
|       |   |
|       |   |___img
|       |   |   |   0.png
|       |   |   |   1.png
```

## Installation
These instructions are for running the Flask server locally.
This assumes you have an Anaconda or miniconda installation functioning. Simplest way to get this running: [https://docs.conda.io/en/latest/miniconda.html]. On Mac download python 3.7 bash installer. Assuming this goes to your Downloads folder, run the following lines in a Terminal shell and follow the prompts to install miniconda.

```
cd Downloads
bash Miniconda3-latest-MacOSX-x86-64.sh
```
After installing open a fresh Terminal and try
```
conda -V
```
This confirms conda is ready to handle our python packages!

The next part assumes you have an SSH key connected to your GitLab account to make pushing/pulling changes easier. Good instructions can be found at [https://docs.gitlab.com/ee/ssh/#generating-a-new-ssh-key-pair]. Generate a key (ed25519 is a good format), and Add it to your Gitlab account. If this is failing you can try the https link from the 'clone' dropdown above in the next section.


To get the tagging tool, the following should be the simplest way to get up and running. Run each line in a Terminal shell, replacing names where applicable.
```
git clone GITHUBURL YOUR_FOLDER_NAME

cd YOUR_FOLDER_NAME/vgac_tagging

conda env create --name YOUR_ENV_NAME --file start_env.yml

conda activate YOUR_ENV_NAME

export FLASK_APP=vgac_tagging

flask run
```


## Pre-processing
Suggested usage:
```
cd pre_processing
parallel --bar --jobs 4 'python yolo_predict_grid_offset.py --game sm3 --dest output --k 5 --grid-size 16 --ui-height 40 --ui-position bot --file {}' ::: PATH/TO/IMAGES/FOR/GAME/*.png
```
Produces csv's and tile sets containing best 5 offsets for each file in folder 'output/sm3/'

```
python greedy_decision.py --pickle-dir output --game sm3 --save-img
```
(the save image flag potentially creates very large matplotlib figures in memory, use with caution)


Reads pickles from output/sm3 and greedily concatenates minimal unique set


Produces sm3_unique_set_NUMTILES.tiles as pickled tile set of encoded pngs (1d ndarray from opencv)


Produces sm3_min_unique_lengths_offsets.csv showing offset decision and local tile set lengths for all images

## Prediction

Example of using a model / classifier to predict affordances in a given screenshot
```
cd affordance_prediction

python predict_image.py --trial-id 10_25_trial.pth --image-path data/validation_img/0.png --output-dir data/validation_output
```

Saves black and white images, with white meaning high probability and black meaning low probability, for each affordance channel.

It treats predictions as probabilities, we have used a cutoff of 0.5 to convert to a binary decision for each affordance. The current model also has a maxpool function with 4x4 window and stride to more resemble the gridded image nature. 

## Background

More info to come after publication

## Usage



## Support

## Contributing
