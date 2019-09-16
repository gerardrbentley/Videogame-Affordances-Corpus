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

To get the tagging tool, the following should be the simplest way to get up and running
```
git clone https://pom-itb-gitlab01.campus.pomona.edu/faim-lab/vgac_tagging.git your_folder_name

cd your_folder_name/tagging_tool

conda env create --name your_env_name --file start_env.yml

conda activate your_env_name

export FLASK_APP=vgac_tagging

flask run
```

## Background

More info to come after publication

## Usage



## Support

## Contributing
