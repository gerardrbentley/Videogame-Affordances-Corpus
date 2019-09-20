# VGAC-Tagging

Current tagging tool for annotating images for VGAC

## Table of Contents

- [Installation](#installation)
- [Background](#background)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)

## Installation
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
git clone git@pom-itb-gitlab01.campus.pomona.edu:faim-lab/vgac_tagging.git YOUR_FOLDER_NAME

cd YOUR_FOLDER_NAME/tagging_tool

conda env create --name YOUR_ENV_NAME --file start_env.yml

conda activate YOUR_ENV_NAME

export FLASK_APP=vgac_tagging

flask run
```

To get the latest dataset go to [https://app.box.com/folder/87149125588] and download the most recent zip file

## Background

More info to come after publication

## Usage



## Support

## Contributing
