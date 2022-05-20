#lattice_processing.py
#Run processing on command line instead of napari. 
#Example for deskewing files in a folder
#python lattice_processing.py --input /home/pradeep/to_deskew --output /home/pradeep/output_save/ --processing deskew
import argparse
from napari_lattice.io import LatticeData,save_tiff
from napari_lattice.utils import read_imagej_roi
from napari_lattice.llsz_core import crop_volume_deskew
from aicsimageio import AICSImage
import os,glob
import pyclesperanto_prototype as cle
from tqdm import tqdm
import dask.array as da


def args_parse():
    """ Parse input arguments"""
    parser = argparse.ArgumentParser(description="Lattice Data processing")
    parser.add_argument('--input',type=str,nargs=1,help="Enter input file")
    parser.add_argument('--output',type=str,nargs=1,help="Enter save folder")
    parser.add_argument('--skew_direction',type=str,nargs=1,help="Enter the direction of skew (default is Y)",default="Y")
    parser.add_argument('--deskew_angle',type=float,nargs=1,help="Enter the agnel of deskew (default is 30)",default=30)
    parser.add_argument('--processing',type=str,nargs=1,help="Enter the processing option: deskew, crop, workflow or workflow_crop")
    parser.add_argument('--roi_file',type=str,nargs=1,help="Enter the path to the ROI file for cropping")
    parser.add_argument('--channel',type=bool,nargs=1,help="If input is a tiff file and there are channel dimensions but no time dimensions, choose as True",default=False)
    parser.add_argument('--voxel_sizes',type=tuple,nargs=1,help="Enter the voxel sizes as (dz,dy,dx). Make sure they are in brackets",default=(0.3,0.1499219272808386,0.1499219272808386))
    parser.add_argument('--file_extension',type=str,nargs=1,help="If choosing a folder, enter the extension of the files (make sure you enter it with the dot at the start, i.e., .czi or .tif), else .czi and .tif files will be used")
    parser.add_argument('--time_range',type=int,nargs=2,help="Enter time range to extract ,example 0 10 will extract first 10 timepoints> default is to extract entire timeseries",default=[0,0])
    parser.add_argument('--channel_range',type=int,nargs=2,help="Enter channel range to extract, default will be all channels. Example 0 1 will extract first two channels. ",default=[0,0])
    args = parser.parse_args()
    return args



def main():
    args = args_parse()
    input_path = args.input[0]
    output_path = args.output[0]+os.sep
    dz,dy,dx = args.voxel_sizes
    deskew_angle = args.deskew_angle
    channel_dimension = args.channel
    skew_dir = args.skew_direction
    processing = args.processing[0].lower() #lowercase
    if processing == "crop":
        roi_file = args.roi_file[0]
        assert roi_file, "Specify roi_file (ImageJ/FIJI ROI Zip file)"
        assert os.path.splitext(roi_file)[1] == ".zip", "ROI file is not a zip file"
            
    time_start,time_end = args.time_range
    channel_start, channel_end = args.channel_range

    print(time_start,time_end)
    print(channel_start, channel_end)
    #Check if input and output paths exist
    assert os.path.exists(input_path), "Cannot find input "+input_path
    assert os.path.exists(output_path), "Cannot find output "+output_path

    if not args.file_extension:
        file_extension = [".czi",".tif",".tiff"]
    else:
        file_extension = args.file_extension

    img_list= []
    #if folder get list of images that match extension in file_extension
    if os.path.isdir(input_path):
        for file_type in file_extension:
            img_list.extend(glob.glob(input_path+os.sep+'*'+file_type))

    #if a single file, just add filename to the image list
    elif os.path.isfile(input_path) and (os.path.splitext(input_path))[1] in file_extension:
        img_list.append(input_path)
    else:
        exit("Do not recognise "+input_path+" as directory or file")
    print(img_list)

    #for img in img_list:  
    img = img_list[0]
    aics_img = AICSImage(img)
    lattice = LatticeData(aics_img,deskew_angle,skew_dir,dx,dy,dz,channel_dimension)

    img_data = lattice.data

    save_name = os.path.splitext(os.path.basename(img))[0]

    if channel_end == 0:
        channel_end = lattice.channels
    if time_end == 0:
        time_end = lattice.time

    if processing == "deskew": 

        save_tiff(vol = img_data,
                    func = cle.deskew_y,
                    time_start = time_start,
                    time_end = time_end,
                    channel_start = channel_start,
                    channel_end = channel_end,
                    save_path = output_path,
                    save_name= save_name,
                    dx = dx,
                    dy = dy,
                    dz = dz,
                    angle = deskew_angle,
                    angle_in_degrees = deskew_angle,
                    voxel_size_x=dx,
                    voxel_size_y=dy,
                    voxel_size_z=dz
                    )
        
    elif processing == "crop":
        roi_list = read_imagej_roi(roi_file)
        for idx, roi_layer in enumerate(tqdm(roi_list, desc="ROI:", position=0)):
            print("Processing ROI "+str(idx)+" of "+str(len(roi_list)))
            deskewed_shape = lattice.deskew_vol_shape
            deskewed_volume = da.zeros(deskewed_shape)
            #Can modify for entering custom z values
            z_start = 0
            z_end = deskewed_shape[0]
            save_tiff(img_data,
                        func = crop_volume_deskew,
                        time_start = time_start,
                        time_end = time_end,
                        channel_start = channel_start,
                        channel_end = channel_end,
                        save_name_prefix  = "ROI_" + str(idx)+"_",
                        save_path = output_path,
                        save_name= save_name,
                        dx = dx,
                        dy = dy,
                        dz = dz,
                        angle = deskew_angle,
                        deskewed_volume=deskewed_volume,
                        roi_shape = roi_layer,
                        angle_in_degrees = deskew_angle,
                        z_start = z_start,
                        z_end = z_end,
                        voxel_size_x=dx,
                        voxel_size_y=dy,
                        voxel_size_z=dz,
                        )
    else:
        exit("Have not implemented "+processing)

if __name__ == '__main__':
    main()