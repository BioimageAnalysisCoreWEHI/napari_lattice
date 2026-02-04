# Installation

This section will guide you through the installation of napari-lattice using conda.
The commands can be copied by clicking the `copy` button at the right side of the respective code blocks below.

## Recommended Installation

We recommend the installation of [Miniforge](https://conda-forge.org/download/) as it is a minimal version of Anaconda Distribution and includes `mamba`, which is faster than conda. 

First, create a new conda environment. To do this, you can look for Miniforge Prompt which we will use for running the commands below:


```bash
mamba create -n napari-lattice -c conda-forge "python==3.11" uv pycudadecon 
```

!!! info
    Python 3.10, 3.11 or 3.12 are supported
    If you do not want deconvolution, you can omit `pycudadecon` from above.

Activate that environment:

```bash
conda activate napari-lattice
```

Then use `uv` for installing the napari-lattice suite using the following 2 commands:

```bash
uv pip install lls-core napari-lattice
```

!!! info

    Using `uv` ensures the installs are fast!


## CLI only

If you do not need to use napari, then you can install just the command line interface only, which has all the features

```bash
mamba create -n napari-lattice -c conda-forge "python==3.11" uv pycudadecon
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
```
