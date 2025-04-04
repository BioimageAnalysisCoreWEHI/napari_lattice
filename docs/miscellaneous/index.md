## Zeiss Lattice Lightsheet 7 

When using the Zeiss LLS7, at the end of every acquisition a maximum intensity projection (MIP) image is created. This can be used for defining the ROIs for cropping. However, the image needs to be transformed so the ROIs are in the right order. 

There are two ways around this:

### Rotate image and then define ROI

- Open the MIP image in Fiji
- Go to Image -> Transform -> Rotate 90 degrees left

    ![fiji_rotate](../images/crop_fiji/001_fiji_rotate.png){ width="300" }

- Wait for the Image to be rotated.
- Once that is finished, draw ROIs using the rectangle tool. 
- Add each ROI to the ROI Manager.
- Save the ROI Manager as a zip file. 
- This ROI file can now be imported into napari-lattice workflows.


### Rotate ROIs

- Alternatively, do not rotate the images, but rotate the ROIs instead.
- To do this, open the Zeiss MIP image.
- Define the ROIs on the data. 
- Run this [Fiji macro](!testhtml) to rotate your ROIs. Save these ROIs as a ROI Manager file (.zip)
