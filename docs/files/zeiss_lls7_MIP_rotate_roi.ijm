/*
 * Fiji macro for rotating ROIS generated on Zeiss LLS7 MIPS for use in napari-lattice
 * If you have drawn ROIs in Fiji using default Zeiss LLS7 MIP, then the ROIs need to be rotated before importing into napari-latice
 * This macro will rotate the ROIs in the ROI Manager by 
 */
 
roiManager("reset");
close("*");

#@ File (style="open",label="Choose LLS7 image") fpath
#@ String(value="The image above will not be opened.\nWe use it to access the metadata", visibility="MESSAGE") hint
#@ File (style="open",label="Choose ROI Manager file") rois_fpath
#@ File (style="directory",label="ROI Save directory") roi_save_dir
fs = File.separator;

//open ROIManager file
roiManager("open", rois_fpath);
nrois = roiManager("count");
if(nrois==0) exit("No ROIS detected");

fname = File.getNameWithoutExtension(rois_fpath);

//check if plugins installed
List.setCommands;
if(List.get("ROIs to Label image")=="") exit("Macro requires BIOP plugin\nPlease install using update sites. Exiting now");
if(List.get("Bounding Box")=="") exit("Macro requires Morphlibj plugin\nPlease install using update sites. Exiting now");

//get series count
run("Bio-Formats Macro Extensions");
Ext.setId(fpath);
Ext.getSeriesCount(seriesCount);

if(seriesCount>1)
{
	seriesNum = getNumber("Detected "+seriesCount+" series. Enter series number", 1);
}
else seriesNum=1;
Ext.setSeries(seriesNum);
Ext.getSizeX(sizeX);
Ext.getSizeY(sizeY)

newImage("test", "8-bit black", sizeX, sizeY, 1);
print("Creating test image with size "+sizeX+"x"+sizeY);
img=getTitle();
selectWindow(img);

Ext.close()

run("Select None");
run("Duplicate...", "duplicate channels=1 frames=1");
img_dup = getTitle();

run("ROIs to Label image");
run("Glasbey on dark");
run("Properties...", "channels=1 slices=1 frames=1 pixel_width=1 pixel_height=1 voxel_depth=1");
close("ROI Manager");
setOption("ScaleConversions", true);
run("Rotate 90 Degrees Left");
rename("mask");
run("Bounding Box", "label=mask show image=mask");
run("To ROI Manager");
close("mask");
close(img);

//visualize rotated rois
///selectWindow(img_dup);
//run("Rotate 90 Degrees Left");
//run("Enhance Contrast", "saturated=0.35");
//roiManager("Show All");
close("mask-BBox");
close("*");


roi_save_path= roi_save_dir+fs+fname+"_ROIs_"+"series_"+seriesNum+"_corrected.zip";
roiManager("Save",roi_save_path);
print("Saved at "+roi_save_path);
exit("New ROIs created for use in napari-lattice.");