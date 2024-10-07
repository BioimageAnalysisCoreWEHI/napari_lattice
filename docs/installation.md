# Installation

## Conda

It is advisable but not required that you install Napari Lattice packages inside a conda environment.
This is because conda makes it much easier to install system complex dependencies.

First, you should work out which version of OpenCL you need:
* Macs with GPUs should use `ocl_icd_wrapper_apple`
* Linux systems with GPUs should use `ocl-icd-system`
* All other systems should use `pocl`

Then you can create a suitable environment, *replacing `pocl` with the appropriate package as necessary*:
```bash
conda install -c conda-forge 'python==3.11.*' pocl
```

Finally, you can proceed with the next step.

## Pip

To install the core package, which includes the Python library and command line interface:

```bash
pip install lls-core
```

To install the Napari plugin:

```bash
pip install napari-lattice
```

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
