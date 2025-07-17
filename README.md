# BRISC - Barcoded Rabies in Situ Connectomics

## Installation

Clone the repository and install it with pip:

```bash
git clone git@github.com:znamlab/brisc.git
cd brisc
pip install .
```

If you want to modify the code, there are a `dev` install option to install the
requirements for `pre-commit`:

```bash
pip install -e ".[dev]"
```


## Get the data

Download the data from XXXX. Unzip and open the `brisc/config.yml` file to change
the file path to the folder where you extracted the data.

### Download external data

For Fig S1, data from the previoulsy published viral libraries must be downloaded and
preprocessed by running brisc/barcode_library_processing/convert_external_libraries.ipynb.

## Generate the figures

The `manuscript_figures` folder contains the notebooks to regenerate all data figures.
In each notebook the `DATA_ROOT` will have to be updated to the path to the folder
where the data is located.
