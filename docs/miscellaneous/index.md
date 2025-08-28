## Zeiss Lattice Lightsheet 7 

When using the Zeiss LLS7, at the end of every acquisition a maximum intensity projection (MIP) image is created. This can be used for defining the ROIs for cropping. However, the ROIs need to be rotated by 90 degrees before it can be used in napari-lattice. 

There are two ways around this in Fiji:
- Rotate the image and then define ROI
- If ROIs have already been defined, rotate the ROIs using a Fiji macro

### 1. Rotate image and then define ROI

Alternatively, you can rotate the image, draw ROIs and save the ROI Manager file.

- Open the MIP image in Fiji
- Go to Image -> Transform -> Rotate 90 degrees left

    ![fiji_rotate](../images/crop_fiji/001_fiji_rotate.png){ width="300" }

- Wait for the Image to be rotated.
- Once that is finished, draw ROIs using the rectangle tool. 
- Add each ROI to the ROI Manager.
- Save the ROI Manager as a zip file. 
- This ROI file can now be imported into napari-lattice workflows.

### 2. Rotate ROIs in Fiji

- Open the LLS7 MIP in Fiji. If you have multiple wells, they will appear as different series.

 ![open_MIP](../images/miscellaneous/001_mips_bioformats.png){ width="300" }

- Draw ROIs on the MIP

 ![annotate_MIP](../images/miscellaneous/002_MIPS_Annotate.png){ width="300" }

- Save the ROIs in a folder
- Download this [Fiji macro](../files/zeiss_lls7_MIP_rotate_roi.ijm). You will need plugins: [BIOP](https://wiki-biop.epfl.ch/en/ipa/fiji/update-site) and [MorpholibJ](https://ijpb.github.io/MorphoLibJ/).
- To run this, drag and drop onto Fiji. Once it opens, you can either click `F5` or `Run -> Run` in the menu on the macro window.
- You will get the following window.

![macro](../images/miscellaneous/003_macro_window.png){ width="500" }

    * `Choose LLS7 image`: Enter path to the image
    * `Choose ROI Manager file`: ROI Manager file created above with areas to crop
    * `ROI Save directory`: Location to save the modified ROIs
- If the MIP has multiple series, then you will get the prompt below.

![series](../images/miscellaneous/004_enter_series.png){ width="300" }

- Specify the series number to process. **Note that you should specify the ROI file for the corresponding series when running this macro.**
- This will process the ROI Manager file and save it in the specified directory with `_corrected` suffix.
- This ROI file can be used in the napari-lattice workflows.



