# napari-lattice

[![License](https://img.shields.io/pypi/l/napari-lattice.svg?color=green)](https://github.com/githubuser/napari-lattice/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-lattice.svg?color=green)](https://pypi.org/project/napari-lattice)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-lattice.svg?color=green)](https://python.org)
[![tests](https://github.com/githubuser/napari-lattice/workflows/tests/badge.svg)](https://github.com/githubuser/napari-lattice/actions)
[![codecov](https://codecov.io/gh/githubuser/napari-lattice/branch/main/graph/badge.svg)](https://codecov.io/gh/githubuser/napari-lattice)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-lattice)](https://napari-hub.org/plugins/napari-lattice)

This napari plugin allows deskewing, cropping, visualisation and designing custom analysis pipelines for lattice lightsheet data, particularly from the Zeiss Lattice Lightsheet. Support will eventually be for other types of data.


## **Documentation**

Check the [Wiki page](https://github.com/BioimageAnalysisCoreWEHI/napari_lattice/wiki) for documentation on how to get started.


*************
## **Features**

<p align="left">
<img src="https://raw.githubusercontent.com/BioimageAnalysisCoreWEHI/napari_lattice/master/resources/LLSZ_window.png" alt="LLSZ_overview" width="500" >
</p>

Functions:
* Deskewing and deconvolution of Zeiss lattice lightsheet images
  * Ability to preview deskewed image at channel or timepoint of interest
* Crop and deskew only a small portion of the image 
* You can import ROIs created in ImageJ into the plugin for cropping
* Create image processing workflows using napari-workflows and apply them to lattice lightsheet data
* Run deskewing, deconvolution and custom image processing workflows from the terminal

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/index.html
-->


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License


=======
Distributed under the terms of the [GPL-3.0 License] license,
"napari_lattice" is free and open source software

## Acknowledgment

 This project was supported by funding from the [Rogers Lab at the Centre for Dynamic Imaging at the Walter and Eliza Hall Institute of Medical Research](https://imaging.wehi.edu.au/). This project has been made possible in part by [Napari plugin accelerator grant](https://chanzuckerberg.com/science/programs-resources/imaging/napari/lattice-light-sheet-data-analysis-toolset/) from the Chan Zuckerberg Initiative DAF, an advised fund of the Silicon Valley Community Foundation.

 Thanks to the developers and maintainers of the amazing open-source plugins such as pyclesperanto, aicsimageio, dask and pycudadecon. ALso, thanks to the imagesc forum!

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GGPL-3.0 License]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
