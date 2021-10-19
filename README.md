# Napari interface for processing Zeiss lattice data

Create conda environment using

    conda create --name <env> --file requirements.txt

The gputools and pyopencl packages need to be installed separately.

Try installing gputools first:

    pip install gputools

If it works, then pypopencl should have been installed and you can start using the environment. 

**If not**, then install pyopencl using [pre-built wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl) here.

For Python 3.9 get files with cp39 in their name. Download the file and then run 

    pip install pyopencl_file_name
Try a file with name cl21 first, if that doesn't work, then try cl12. Install gputools after this using pip. 

Clone the repo or download the package. Open a terminal from within.
Run the package:

    python main.py

This will start a napari instance.
The worfklow currently is:
* Open a czi file (skewed raw data)
* Deskew a single timepoint
* Use the deskewed stack and max projection for previewing and cropping an area of interest

![Preview of widget](resources\preview_video.mp4)


