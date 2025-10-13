# BRISC - Barcoded Rabies in Situ Connectomics

## Installation

This should work with any python > 3.8 and was tested with python version up to 3.11.
To install clone the repository and install it with pip:

```bash
git clone git@github.com:znamlab/brisc.git
cd brisc
pip install .
```

If you want to modify the code, there are a `dev` install option to install the
requirements for `pre-commit`:

```bash
pip install -e ".[dev]"
pre-commit install
```


## Get the data

Download the data from [figshare](https://figshare.com/s/dd23702b49abb37f7ba0?file=56354084). Unzip and open the `brisc/config.yml` file to change
the file path to the folder where you extracted the data.

### Download external data

For Fig S1, data from the previously published viral libraries must be downloaded and
preprocessed by running brisc/barcode_library_processing/convert_external_libraries.ipynb.

## Generate the figures

The `manuscript_figures` folder contains the notebooks to regenerate all data figures.
In each notebook the `DATA_ROOT` will have to be updated to the path to the folder
where the data is located.

## System requirements:

Tested on:

Operating System: Rocky Linux 8.7 (Green Obsidian)

Architecture: x86-64

A full software dependency list with version numbers is available in package_list.txt

Expected installation time: ~10 minutes

Required non-standard hardware:

Figure 2 requires a minimum of 64GB RAM to reproduce overview images

Expected run time per figure:

Fig. 1. ~ 2 minutes

Fig. 2. ~ 10 minutes

Fig. 3. ~ 10 minutes

Fig. 4. ~ 20 minutes

Fig. 5. ~ 20 minutes

Fig. 6. ~ 20 minutes

Supp Fig. 1. ~ 2 minutes

Supp Fig. 3. ~ 10 minutes

Supp Fig. 4. ~ 10 minutes

Supp Fig. 6. ~ 10 minutes
