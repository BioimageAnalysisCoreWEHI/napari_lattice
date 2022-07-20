# napari-lattice

[![License](https://img.shields.io/pypi/l/napari-lattice.svg?color=green)](https://github.com/githubuser/napari-lattice/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-lattice.svg?color=green)](https://pypi.org/project/napari-lattice)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-lattice.svg?color=green)](https://python.org)
[![tests](https://github.com/githubuser/napari-lattice/workflows/tests/badge.svg)](https://github.com/githubuser/napari-lattice/actions)
[![codecov](https://codecov.io/gh/githubuser/napari-lattice/branch/main/graph/badge.svg)](https://codecov.io/gh/githubuser/napari-lattice)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-lattice)](https://napari-hub.org/plugins/napari-lattice)

This napari plugin allows deskewing, cropping, visualisation and analysis of lattice lightsheet data skewed in the Y plane (Zeiss Lattice Lightsheet). Support will eventually be for other types of data.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/index.html
-->

## **Installation**


1. Use a conda environment for installation. You will need [Anaconda Navigator](https://www.anaconda.com/products/individual) or a lighter version [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. Click on the Anaconda prompt or terminal and create an environment first:

    conda create -n napari-lattice python=3.9

2. You can use any name instead of "llsz". Once an environment is created, activate it by typing with the name you used:

        conda activate napari-lattice

3. Start with installing [pyopencl](https://documen.tician.de/pyopencl/)

        conda install -c conda-forge pyopencl

    If you have trouble installing pyopencl on Windows, use a precompiled wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl). As we use python 3.9, you have to download and try the wheels which have cp39 in their name. For example, if we download `pyopencl‑2022.1‑cp39‑cp39‑win_amd64.whl`, then we navigate to the download folder and run:

        pip install pyopencl‑2022.1‑cp39‑cp39‑win_amd64.whl

    If this version doesn't work, try `pyopencl‑2021.2.9+cl12‑cp39‑cp39‑win_amd64.whl`

4. You need to install [napari](https://pypi.org/project/napari/) first. Once napari is installed, to install latest version of the napari_lattice plugin, by typing the following command in the terminal :

        pip install git+https://github.com/BioimageAnalysisCoreWEHI/napari_lattice.git

The plugin will be made available by napari-> Install/Uninstall Plugins soon.

*************
## **Features**

<p align="left">
<img src="https://raw.githubusercontent.com/BioimageAnalysisCoreWEHI/napari_lattice/master/resources/LLSZ_window.png" alt="LLSZ_overview" width="500" >
</p>
All transformations are now powered by clesperanto.

Functions:
* Deskewing of Zeiss lattice lightsheet images
  * Ability to preview deskewed image at channel or timepoint of interest
* Crop and deskew only a small portion of the image 
* You can import ROIs created in ImageJ into the plugin for cropping
* Create image processing workflows using napari-workflows and apply them to lattice data
* Run napari_lattice from the terminal

### Deconvolution not implemented yet!

*****
## **Usage**

Once installed, just [start napari](https://napari-staging-site.github.io/tutorials/fundamentals/getting_started.html#command-line-usage) and the plugin should be under "Plugins" tab

Note that channels and timepoints start from 0.

### ****File compatibility**

* You can directly open a Zeiss lattice file by dragging the file into napari. 
* You will get a prompt to `Choose Reader`. Select `aicsimageio-out-of-memory` for large datasets that don't fit in memory. 
* If the channels are split into different layers, press `Shift` and select all the channels. Right click on the highlighted layers and then click `Merge to stack`. 
* Once image has loaded, click `Initialize Plugin`. If its a czi file, it will load voxel size from the metadata. Once the button turns green, it means the plugin is initialized

### **Deskewing**

To preview the deskewed output of a single timepoint and channel, choose the layer under the box `img data`, enter the timepoint and channel of interest and click  `Preview`. You can see the status of the processing in the terminal. 
Once finished, you will see a deskewed image and the corresponding maximum intensity projection as two layers.

<p align="left">
<img src="https://raw.githubusercontent.com/BioimageAnalysisCoreWEHI/napari_lattice/master/resources/preview_deskew.png" alt="Deskew Preview" width="500" >
</p>

To deskew and save a range of timepoints and channels, select the `Deskew` tab. Choose the channel and time ranges, save directory and click `Save`.


### **Cropping**

* Click on the `Crop & Deskew` tab. 
* Clicking on `Click to activate Cropping Preview` will activate the `Import ImageJ ROI` and `Crop Preview` buttons. This will also add a shapes layer for the ROIs or shapes. 

Crop & Deskew (initial)            |  Crop & Deskew (Activated) 
:-------------------------:|:-------------------------:
![initial](https://raw.githubusercontent.com/BioimageAnalysisCoreWEHI/napari_lattice/master/resources/crop%26deskew_initial.png)  |  ![active](https://raw.githubusercontent.com/BioimageAnalysisCoreWEHI/napari_lattice/master/resources/crop%26deskew_active.png)

* For cropping regions of interest on the deskewed image, you can either:
  * draw the regions using shapes layer in napari AND/OR
  * import ImageJ ROIs an ImageJ ROI file (.zip). 
* You can use a combination of shapes and ImageJ ROIs if needed.
* To preview the cropped image, select the ROI and then click `Crop Preview`.

Note: The cropping functionality works by finding the inverse transform of the ROI from the deskewed image and extracting the corresponding data from the skewed or raw image. Only the extracted portion will be deskewed. This saves time and is low on memory compared to deskewing the entire image and then cropping. You can select the range of channels and time poitns you'd like to crop and save under `Crop and Save Data` section.

### **Workflow**

This section uses [napari-workflows](https://github.com/haesleinhuepf/napari-workflows) to implement custom image processing routines on lattice lightsheet datasets. You can create a custom workflow using [napari-assistant](https://www.napari-hub.org/plugins/napari-assistant) and save it. This can then be loaded into the napari_lattice plugin under the `Workflow` tab which can then be applied to the deskewed output.

![workflow](https://raw.githubusercontent.com/BioimageAnalysisCoreWEHI/napari_lattice/master/resources/workflow.png) 

If you'd like to use the shapes layer for cropping and applying a workflow, tick `Crop Data` checkbox. This will crop the data and then apply the workflow. 

You can also use custom python functions with workflows. The function needs to be called within the workflow yml file and the custom function should be a `.py` file in the same directory as the workflow file. An example will be added soon. 

### More details will be added soon.

*******

## **Batch processing (No GUI)**

The software can also be used to batch process data in a folder. To run the program, you will have to open a terminal and make sure your environment is activated using `conda activate`. To see a list of options, you type `napari_lattice -h` . The `-h` means help and will print out a list of options on how to run the program. Currently, the available options are:

    usage: napari_lattice [-h] [--input INPUT] [--output OUTPUT] [--skew_direction SKEW_DIRECTION] [--deskew_angle DESKEW_ANGLE] [--processing PROCESSING] [--roi_file ROI_FILE] [--channel CHANNEL]
                        [--voxel_sizes VOXEL_SIZES] [--file_extension FILE_EXTENSION] [--time_range TIME_RANGE TIME_RANGE] [--channel_range CHANNEL_RANGE CHANNEL_RANGE][--workflow_path WORKFLOW_PATH]

    Lattice Data processing

    optional arguments:
    -h, --help            show this help message and exit
    --input INPUT         Enter input file or folder of files
    --output OUTPUT       Enter save folder
    --skew_direction SKEW_DIRECTION
                            Enter the direction of skew (default is Y)
    --deskew_angle DESKEW_ANGLE
                            Enter the angle of deskew (default is 30)
    --processing PROCESSING
                            Enter the processing option: deskew, crop, workflow or workflow_crop
    --roi_file ROI_FILE   Enter the path to the ROI file or a folder of ROI files for cropping
    --channel CHANNEL     If input is a tiff file and there are channel dimensions but no time dimensions, choose as True
    --voxel_sizes VOXEL_SIZES
                            Enter the voxel sizes as (dz,dy,dx). Make sure they are in brackets. If its a czi file, will read it from metadata.
    --file_extension FILE_EXTENSION
                            If choosing a folder, enter the extension of the files (make sure you enter it with the dot at the start, i.e., .czi or .tif), else .czi and .tif files will be used
    --time_range TIME_RANGE TIME_RANGE
                            Enter time range to extract ,example 0 10 will extract first 10 timepoints> default is to extract entire timeseries if no range is specified
    --channel_range CHANNEL_RANGE CHANNEL_RANGE
                            Enter channel range to extract, default will be all channels if no range is specified. Example 0 1 will extract first two channels.
    --workflow_path WORKFLOW_PATH
                            If using workflow or workflow_crop, enter the path to the workflow file (*.yml)

As an example, if you would like to deskew a bunch of files in a folder, you would need to define the `input folder`, `save location` and `processing` option (deskew). The skew direction by default is `Y` and `deskew angle` is 30.

If the input folder is:

    C:\source

and output or save location is:

    C:\deskewed

then you run the command, 

    napari_lattice --input "C:\source" --output "C:\deskewed" --processing deskew

Note that the folder locations are in quotes. This is useful especially if you have spaces in the folder names. 

If you'd like to run cropping and deskew and have a list of ImageJ ROI files (.zip) files which are located at, say `D:\rois` and the folder locations are the same as above, then we can use the command:

    napari_lattice --input "C:\source" --output "C:\deskewed" --roi_file "D:\rois" --processing crop

***Note: The ROI files need to have the same name as the image files.***

You can also replace the `input`, `output`, and `roi_file` arguments with file locations instead of folders.

Please find sample data for testing in the `sample_data` folder above

***Data Credit: Cindy Evelyn & Niall Geoghegan, Walter and Eliza Hall Institute of Medical Research, Melbourne, Australia***



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License


=======
Distributed under the terms of the [GPL-3.0 License] license,
"napari_lattice" is free and open source software

## Acknowledgment

 This project was supported by funding from the [Rogers Lab at the Centre for Dynamic Imaging at the Walter and Eliza Hall Institute of Medical Research](https://imaging.wehi.edu.au/). This project has been made possible in part by [Napari plugin accelerator grant](https://chanzuckerberg.com/science/programs-resources/imaging/napari/lattice-light-sheet-data-analysis-toolset/) from the Chan Zuckerberg Initiative DAF, an advised fund of the Silicon Valley Community Foundation.

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
