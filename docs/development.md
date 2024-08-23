# Development

## Structure
The repo is divided into two python packages: `core` (the core processing and CLI) and `plugin` (the Napari GUI).
Within each directory is:

* The python package
* `pyproject.toml`, the package metadata such as the dependencies, and 
* `tests/`, which contains the tests

## Installation

For local development, first clone the repo and then run the following from the repository root:
```bash
pip install -e core -e plugin
```

## Technologies

### Pydantic

Used for defining the parameter sets, performing parameter validation and conversion. These models live in `core/lls_core/models`.
Note, `lls_core` uses Pydantic 1.X, which has a different API to Pydantic 2.X.
You can find relevant documentation here: <https://docs.pydantic.dev/1.10/>

### Xarray

Used for all image representations, where they are treated as multidimensional arrays with dimensional labels (X, Y, Z etc).
Refer to: <https://xarray.pydata.org/.>

### Typer

The CLI is defined using Typer: <https://typer.tiangolo.com/.>

### magicgui and magicclass

These packages are used to define the GUI, which you can find in `plugin/napari_lattice`.
[`magicclass`](https://hanjinliu.github.io/magic-class/) builds on [`magicgui`](https://pyapp-kit.github.io/magicgui/) by providing the `@magicclass` decorator which turns a Python class into a GUI.

## Adding a new parameter

Whenever a new parameter is added, the following components need to be updated:

* Add the parameter to the Pydantic models
* Add the parameter to the CLI (`core/lls_core/cmds/__main__.py`), and define mapping between CLI and Pydantic using the `CLI_PARAM_MAP`
* Add the field to the GUI in `plugin/napari_lattice/fields.py`
* Define the new processing logic in `core/lls_core/models/lattice_data.py`

An example of this can be found in this commit: <https://github.com/BioimageAnalysisCoreWEHI/napari_lattice/pull/47/commits/16b28fec307f19e73b8d55e677621082037b2710>.

## Testing

The tests are run using [pytest](https://docs.pytest.org/en/7.4.x/).
To install the testing dependencies, use `pip install -e 'core[testing]' -e 'plugin[testing]'`
Since there are two separate packages, you will have to specify the location of each test directory.
To run all the tests, use `pytest core/tests/ plugin/tests` from the root directory.

## Documentation

Docs are built with [mkdocs](https://www.mkdocs.org/).

To modify the docs, you need the docs dependencies, so clone the repo and then:

```bash
pip install -e 'core[docs]'
```

The key files are:
    
    * `mkdocs.yml`, which is the main config file for mkdocs, and
    * `docs/` which is a directory containing markdown files. Each new file that gets added there will create a new page in the website.

Some useful `mkdocs` commands:

* `mkdocs serve` runs the development server which hosts the docs on a local web server. Any changes to your markdown files will be reflected in this server, although you sometimes have to restart the server if you make a change to configuration
* `mkdocs gh-deploy` builds the docs, and pushes them to GitHub Pages. This updates the docs at <https://bioimageanalysiscorewehi.github.io/napari_lattice/>.
