# Installation

This section will guide you through the installation of napari-lattice using conda.
The commands can be copied by clicking the `copy` button at the right side of the respective code blocks below.

## Recommended Installation

We recommend the installation of [Miniconda](https://docs.anaconda.com/miniconda/install/) as is a minimal version of Anaconda Distribution. 

??? Tip
    If you can [install `mamba` in your environment](https://stackoverflow.com/a/76765625), the conda installation commands will be much faster. Once configured correctly, just replace `conda` with `mamba` in the commands below.

First, create a new conda environment:


```bash
conda create -n napari-lattice -c conda-forge "python==3.10" uv pycudadecon "numpy<2"
```

!!! info

    We include `pycudadecon` to enable GPU accelerated deconvolution.

Activate that environment:

```bash
conda activate napari-lattice
```

Then use `uv` to quickly install the napari-lattice suite using the following 2 commands:

```bash
uv pip install lls-core napari-lattice
```

```bash
uv pip install --upgrade aicsimageio "napari==0.5.5" "numpy<2"
```

## CLI only

If you do not need to use napari, then you can install just the command line interface only, which has all the features

```bash
conda create -n napari-lattice -c conda-forge "python==3.10" uv pycudadecon "numpy<2"
```

```bash
uv pip install lls-core
```


## Development Installation

To install the development version of `lls-core`, create the `napari-lattice` environment as above, but instead of installing from pip in the last step:

```bash
git clone https://github.com/BioimageAnalysisCoreWEHI/napari_lattice.git
cd napari_lattice
uv pip install -e core -e plugin
uv pip install napari --upgrade "numpy<2"
```
