//macro to rotate ROIs from a Zeiss LLS7 MIP by 90 degree to the left
//This allows importing into napari-lattice in the right orientation

List.setCommands;
ptbiop_check = List.get("ROIs to Label image");
morpholibj_check = List.get("Bounding Box");
if(ptbiop_check=="") exit("Cannot proceed without activating update sites: PTBIOP");
if(morpholibj_check=="") exit("Cannot proceed without activating update sites: IJPB plugins");

// needs BIOP and IJPB Plugins (morpholibJ)
close("ROI Manager");
run("Close All");
#@ File (style="open",label="Open CZI file") fpath
#@ File (style="open",label="Open ROI Manager with crops (zip)") rois_fpath

//fpath  = "Z:\\LLS\\LLSZ\test.czi";
//rois_fpath = "Z:\\LLS\\LLSZ\\test.zip";

short_rois = substring(rois_fpath,0,indexOf(rois_fpath,".zip"));

run("Bio-Formats Importer", "open=["+fpath+"] color_mode=Default rois_import=[ROI manager] specify_range view=Hyperstack stack_order=XYCZT c_begin=1 c_end=1 c_step=1 t_begin=1 t_end=1 t_step=1");
roiManager("Open",rois_fpath);
run("ROIs to Label image");
run("Glasbey on dark");
run("Properties...", "channels=1 slices=1 frames=1 pixel_width=1 pixel_height=1 voxel_depth=1");
close("\\Others");
close("ROI Manager");
setOption("ScaleConversions", true);
run("Rotate 90 Degrees Left");
rename("mask");
run("Bounding Box", "label=mask show image=mask");
run("To ROI Manager");
roiManager("Save",short_rois+"_corrected.zip");