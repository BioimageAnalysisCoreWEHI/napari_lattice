# llsz_napari

[![License](https://img.shields.io/pypi/l/llsz_napari.svg?color=green)](https://github.com/pr4deepr/llsz_napari/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/llsz_napari.svg?color=green)](https://pypi.org/project/llsz_napari)
[![Python Version](https://img.shields.io/pypi/pyversions/llsz_napari.svg?color=green)](https://python.org)
[![tests](https://github.com/pr4deepr/llsz_napari/workflows/tests/badge.svg)](https://github.com/pr4deepr/llsz_napari/actions)
[![codecov](https://codecov.io/gh/pr4deepr/llsz_napari/branch/main/graph/badge.svg)](https://codecov.io/gh/pr4deepr/llsz_napari)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/llsz_napari)](https://napari-hub.org/plugins/llsz_napari)

LLSZ

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

Use a conda environment. Start with installing [pyopencl](https://documen.tician.de/pyopencl/)

    conda install -c conda-forge pyopencl

To install latest development version :

    pip install git+https://github.com/pr4deepr/llsz_napari.git@napari_plugin


#### [**Sample data for testing**](https://cloudstor.aarnet.edu.au/plus/s/700eD6EcgOODovI) 
***Credit: Cindy Evelyn & Niall Geoghegan, Walter and Eliza Hall Institute of Medical Research, Melbourne, Australia***

Once installed, just start napari as normal and the plugin should be under "Plugins" tab

Functions:
* Open a czi lattice file
* Preview deskewing on any time or channel
* Preview cropping on any time or channel
* Save deskewed stack for time and channel range of interest
* Save cropped stack for time and channel range of interest

![image](/resources/LLSZ_window.png)

This plugin uses gputools or dask for affine transformation.
API for transformations are defined in /src/llsz/transformations.py. However, most of this is being implemented in pyclesperanto to simplify the code


To do:
* Clean up UI (Menu options?)
* Add pyclesperanto as backend for transformations
* Implement image analysis workflow option
* Include deconvolution
* Add batch processing option (no napari -> magic-class or magicgui only)
* Document functions consistently


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"llsz_napari" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/pr4deepr/llsz_napari/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
