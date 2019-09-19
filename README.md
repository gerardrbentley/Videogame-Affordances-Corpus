# VGAC-Tagging

Current tagging tool for annotating images for VGAC

## Table of Contents

- [Installation](#installation)
- [Background](#background)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)

## Installation
This assumes you have an Anaconda or miniconda installation functioning. Simplest way to get this running: [https://docs.conda.io/en/latest/miniconda.html]. On Mac download python 3.7 bash installer. Assuming this goes to your Downloads folder, do the following in a Terminal shell and follow the prompts. After installing open a fresh Terminal and try `conda -V`

```
cd Downloads
bash Miniconda3-latest-MacOSX-x86-64.sh
```

The next part assumes you have an SSH key connected to your GitLab account to make pushing/pulling changes easier. Good instructions can be found at [https://docs.gitlab.com/ee/ssh/]. If this is failing try the https link from the 'clone' dropdown above.


To get the tagging tool, the following should be the simplest way to get up and running
```
git clone git@pom-itb-gitlab01.campus.pomona.edu:faim-lab/vgac_tagging.git your_folder_name

cd your_folder_name/tagging_tool

conda env create --name your_env_name --file start_env.yml

conda activate your_env_name

export FLASK_APP=vgac_tagging

flask run
```

To get the latest dataset go to [https://app.box.com/folder/87149125588] and download the most recent zip file

## Background

More info to come after publication

## Usage



## Support

## Contributing
