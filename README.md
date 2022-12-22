# napari-lattice

[![License](https://img.shields.io/pypi/l/napari-lattice.svg?color=green)](https://github.com/githubuser/napari_lattice/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-lattice.svg?color=green)](https://pypi.org/project/napari_lattice)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-lattice.svg?color=green)](https://python.org)
[![tests](https://github.com/BioimageAnalysisCoreWEHI/napari_lattice/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/BioimageAnalysisCoreWEHI/napari_lattice/actions/workflows/test_and_deploy.yml)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/napari-lattice)](https://pypistats.org/packages/napari-lattice)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-lattice)](https://napari-hub.org/plugins/napari-lattice)

This napari plugin allows deskewing, cropping, visualisation and designing custom analysis pipelines for lattice lightsheet data, particularly from the Zeiss Lattice Lightsheet. The plugin has also been otpimixed to run in headless mode.


## **Documentation**

Check the [Wiki page](https://github.com/BioimageAnalysisCoreWEHI/napari_lattice/wiki) for documentation on how to get started.


*************


<p align="left">
<img src="https://raw.githubusercontent.com/BioimageAnalysisCoreWEHI/napari_lattice/master/resources/LLSZ_window.png" alt="LLSZ_overview" width="500" >
</p>

**Functions**

* Deskewing and deconvolution of Zeiss lattice lightsheet images
  * Ability to preview deskewed image at channel or timepoint of interest
* Crop and process only a small portion of the image 
* Import ImageJ ROIs for cropping
* Create image processing workflows using napari-workflows
* Run deskewing, deconvolution and custom image processing workflows from the terminal
* Files can be saved as h5 (BigDataViewer/BigStitcher) or tiff files
* Run in terminal without napari, enabling processing workflows on the HPC

**Key Features**

Apply custom image processing workflows using `napari-workflows`. 
* [Interactive workflow generation (no coding experience needed)](https://github.com/BioimageAnalysisCoreWEHI/napari_lattice/wiki/5.-Workflows-(Interactive:-no-coding)#workflow)
* [Use custom python functions/modules within workflows](https://github.com/BioimageAnalysisCoreWEHI/napari_lattice/wiki/5.1-Workflows-(Custom-workflow))
* [How to use Cellpose for cell segmentation](https://github.com/BioimageAnalysisCoreWEHI/napari_lattice/wiki/5.1-Workflows-(Custom-workflow)#cellpose)


Support will be added for more file formats in the future.

Sample lattice lightsheet data download: https://doi.org/10.5281/zenodo.7117784

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

Distributed under the terms of the [GPL-3.0 License] license,
"napari_lattice" is free and open source software

## Acknowledgment

 This project was supported by funding from the [Rogers Lab at the Centre for Dynamic Imaging at the Walter and Eliza Hall Institute of Medical Research](https://imaging.wehi.edu.au/). This project has been made possible in part by [Napari plugin accelerator grant](https://chanzuckerberg.com/science/programs-resources/imaging/napari/lattice-light-sheet-data-analysis-toolset/) from the Chan Zuckerberg Initiative DAF, an advised fund of the Silicon Valley Community Foundation.

 Thanks to the developers and maintainers of the amazing open-source plugins such as [pyclesperanto](https://github.com/clEsperanto/pyclesperanto_prototype), [aicsimageio](https://github.com/AllenCellModeling/aicsimageio), [dask](https://github.com/dask/dask) and [pycudadecon](https://github.com/tlambert03/pycudadecon).
 Thanks in particular to the developers of open source projects: [LLSpy](https://github.com/tlambert03/LLSpy) and [lls_dd](https://github.com/VolkerH/Lattice_Lightsheet_Deskew_Deconv) as they were referred to extensively for developing napari-lattice.
 Thanks to the imagesc forum!

## Issues

If you encounter any problems, please [file an issue](https://github.com/BioimageAnalysisCoreWEHI/napari_lattice/issues) along with a detailed description.

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
