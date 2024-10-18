# Installation

It is advisable but not required that you install Napari Lattice packages inside a conda environment.
This is because conda makes it much easier to install system complex dependencies.

To install the core package, which includes the Python library and command line interface:

```bash
pip install lls-core
```

To install the Napari plugin:

```bash
pip install napari-lattice
```

## Faster Installation using `uv`

Installing both `lls-core` and `napari-lattice` takes about 10 minutes using a fast internet connection.
If your connection is slow, or you just don't want to wait this long, you can install it in about 10 seconds using `uv`:

1. Install `uv` following [the instructions here](https://docs.astral.sh/uv/getting-started/installation/)
2. Create a new project using `uv init my-project`
3. `uv add lls-core napari-lattice`

## Recommended Plugins

If you are working with `.czi` files, it is recommended that you install the [`napari-aicsimageio`](https://github.com/AllenCellModeling/napari-aicsimageio) plugin for Napari.
`napari-lattice` does not depend on this plugin, so you will have to install it separately, either using the Napari plugin manager, or using `pip install napari-aicsimageio`.

## CUDA Deconvolution

If you have access to a CUDA-compatible GPU, you can enable deconvolution using `pycudadecon`.

If you're using conda (or micromamba etc), you can run the following:

```bash
conda install -c conda-forge pycudadecon
```

Otherwise, you will have to manually ensure the systems dependencies are installed, and then:

```bash
pip install lls-core[deconvolution]
```

## Development Versions

To install the development version of `lls-core`:

```bash
pip install git+https://github.com/BioimageAnalysisCoreWEHI/napari_lattice.git#subdirectory=core
```

For `napari-lattice`:

```bash
pip install git+https://github.com/BioimageAnalysisCoreWEHI/napari_lattice.git#subdirectory=plugin
```
