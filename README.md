# BRISC - Barcoded Rabies in Situ Connectomics

## Installation

This requires python >= 3.10 and was tested with python versions up to 3.14.

### Using uv (recommended)

Clone the repository and let [uv](https://docs.astral.sh/uv/) create the
environment from the pinned `uv.lock`:

```bash
git clone git@github.com:znamlab/brisc.git
cd brisc
uv sync --extra figures
```

This installs an exact, reproducible set of dependencies (including the other
znamlab packages, pulled directly from GitHub) into a local `.venv`. Use
`uv run jupyter lab` to launch Jupyter inside that environment, or prefix any
command with `uv run` to execute it there. Add `--extra dev` as well if you want
to modify the code and need the `pre-commit`/`ruff`/`pytest` tooling.

### Using pip

Alternatively, clone the repository and install it with pip:

```bash
git clone git@github.com:znamlab/brisc.git
cd brisc
pip install ".[figures]"
```

The `.[figures]` will install jupyter and ipykernel to run the notebooks used to
generate figures. Use the plain `pip install .` for a minimal installation.

If you want to modify the code, there are a `dev` install option to install the
requirements for `pre-commit`:

```bash
pip install -e ".[dev]"
pre-commit install
```


## Get the data

Download the data from [figshare](https://figshare.com/s/dd23702b49abb37f7ba0?file=56354084)
and unzip it. The unzipped folder contains a `config.yml` at its top level
(e.g. `<extracted_folder>/config.yml`) with contents like:

```yaml
data_root:
  processed: /path/to/extracted_folder
  raw: /path/to/extracted_folder
```

Open that file and set both `data_root.processed` and `data_root.raw` to the
absolute path of the folder you extracted the data into (the same folder that
contains this `config.yml`). This is what lets `flexiznam` resolve data paths
when a notebook's `DATA_ROOT` is set to that folder.

### Download external data

For Fig 1f, data from previously published viral libraries must be downloaded and
preprocessed by running brisc/barcode_library_processing/convert_external_libraries.ipynb.

## Generate the figures

The `manuscript_figures` folder contains the notebooks to regenerate all data figures.
In each notebook the `DATA_ROOT` will have to be updated to the path to the folder
where the data is located.


### Measured run time and peak RAM per notebook

The table below was measured by running each notebook end-to-end with
`jupyter nbconvert --execute` inside the `uv`-managed environment, one
notebook at a time.


Measured on a MacBook (Apple M1 Pro, 8 cores, 16GB RAM) using data on an
external drive.

| Notebook | Status | Run time | Peak RAM |
|---|---|---|---|
| figure1_plasmid_barcoding_schema_library | ✅ | 5 min | 5.0 GB |
| figure2_data_overview_images | ✅ | 11 min | 9.0 GB |
| figure3_barcodes_in_cells_overview | ✅ | 3 min | 4.6 GB |
| figure4_spatial_barcodes | ✅ | 7 min | 5.4 GB |
| figure5_connectivity_matrices | ✅ | 12 min | 5.5 GB |
| figure6_long_range | ✅ | 7 min | 6.4 GB |
| print_numbers | ✅ | 18 s | 0.9 GB |
| suppfig2_diversity | ✅ | 2 min | 4.4 GB |
| suppfig4_barcodelength | ✅ | 4 min | 0.7 GB |
| suppfig5_mcherry_cellpositions | ✅ | 3 min | 3.1 GB |
| suppfig6_transcriptomics_validation | ✅ | 4 min | 4.9 GB |
| suppfig8_multiple_starter_bcs | ✅ | 5 min | 6.8 GB |
| suppfig9_double_labeling_analysis | ✅ | 4 min | 5.1 GB |
| suppfig_reviewer_elevation | ✅ | 6 min | 6.1 GB |
