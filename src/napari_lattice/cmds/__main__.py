#lattice_processing.py
#Run processing on command line instead of napari. 
#Example for deskewing files in a folder
#python lattice_processing.py --input /home/pradeep/to_deskew --output /home/pradeep/output_save/ --processing deskew
import argparse,os,glob,sys
from napari_lattice.io import LatticeData, save_img, save_img_workflow
from napari_lattice.utils import read_imagej_roi, get_all_py_files, get_first_last_image_and_task,modify_workflow_task
from napari_lattice.llsz_core import crop_volume_deskew
from aicsimageio import AICSImage
import pyclesperanto_prototype as cle
from tqdm import tqdm
import dask.array as da
from napari_workflows import Workflow, WorkflowManager
from napari_workflows._io_yaml_v1 import load_workflow
from pathlib import Path
from .. import config

from ..ui_core import _read_psf

#define parser class so as to print help message
class ArgParser(argparse.ArgumentParser): 
   def error(self, message):
      sys.stderr.write('error: %s\n' % message)
      self.print_help()
      sys.exit(2)

#TODO: Implement deconvolution

def args_parse():
    """ Parse input arguments"""
    parser = argparse.ArgumentParser(description="Lattice Data processing")
    parser.add_argument('--input',type=str,nargs=1,help="Enter input file", required=True)
    parser.add_argument('--output',type=str,nargs=1,help="Enter save folder", required=True)
    parser.add_argument('--skew_direction',type=str,nargs=1,help="Enter the direction of skew (default is Y)",default="Y")
    parser.add_argument('--deskew_angle',type=float,nargs=1,help="Enter the deskew angle (default is 30)",default=30.0)
    parser.add_argument('--processing',type=str,nargs=1,help="Enter the processing option: deskew, crop, workflow or workflow_crop", required=True)
    parser.add_argument('--deconvolution',type=str,nargs=1,help="Specify the device to use for deconvolution. Options are cpu or cuda_gpu")
    parser.add_argument('--deconvolution_num_iter',type=int,nargs=1,help="Enter the number of iterations to run Richardson-Lucy deconvolution (default is 10)")
    parser.add_argument('--deconvolution_psf',type=str,nargs="+",help="Enter paths to psf file/s separated by commas or you can enter each path with double quotes") #use + for nargs for flexible no of args
    parser.add_argument('--roi_file',type=str,nargs=1,help="Enter the path to the ROI file for performing cropping (only valid for -processing where crop or workflow_crop is specified")
    parser.add_argument('--voxel_sizes',type=tuple,nargs=1,help="Enter the voxel sizes as (dz,dy,dx). Make sure they are in brackets",default=(0.3,0.1499219272808386,0.1499219272808386))
    parser.add_argument('--file_extension',type=str,nargs=1,help="If choosing a folder, enter the extension of the files (make sure you enter it with the dot at the start, i.e., .czi or .tif), else .czi and .tif files will be used")
    parser.add_argument('--time_range',type=int,nargs=2,help="Enter time range to extract, default will be entire timeseries if no range is specified. For example, 0 9 will extract first 10 timepoints")
    parser.add_argument('--channel_range',type=int,nargs=2,help="Enter channel range to extract, default will be all channels if no range is specified. For example, 0 1 will extract first two channels. ")
    parser.add_argument('--workflow_path',type=str,nargs=1,help="Enter path to the workflow file '.yml")
    parser.add_argument('--output_file_type',type=str,nargs=1,help="Save as either tif or h5, defaults to tif")
    parser.add_argument('--channel',type=bool,nargs=1,help="If input is a tiff file and there are channel dimensions but no time dimensions, choose as True",default=False)
    args = parser.parse_args()
    return args




def main():
    args = args_parse()
    print(args)
    input_path = args.input[0]
    output_path = args.output[0]+os.sep
    dz,dy,dx = args.voxel_sizes
    deskew_angle = args.deskew_angle
    channel_dimension = args.channel
    skew_dir = args.skew_direction
    processing = args.processing[0].lower() #lowercase


    if processing == "crop" or processing == "workflow_crop":
        assert args.roi_file, "Specify roi_file (ImageJ/FIJI ROI Zip file)"
        roi_file = args.roi_file[0]
        if os.path.isfile(roi_file): #if file make sure it is a zip file
            assert os.path.splitext(roi_file)[1] == ".zip", "ROI file is not a zip file"
    


    #print(time_start,time_end)
    #print(channel_start, channel_end)
    #Check if input and output paths exist
    assert os.path.exists(input_path), "Cannot find input "+input_path
    assert os.path.exists(output_path), "Cannot find output "+output_path

    if not args.file_extension:
        file_extension = [".czi",".tif",".tiff"]
    else:
        file_extension = args.file_extension
    
    #Initialise list of images and ROIs
    img_list= []
    roi_list= []

    #If input_path a directory, get a list of images
    if os.path.isdir(input_path):
        for file_type in file_extension:
            img_list.extend(glob.glob(input_path+os.sep+'*'+file_type))
        print("List of images: ", img_list)
    elif os.path.isfile(input_path) and (os.path.splitext(input_path))[1] in file_extension:
        img_list.append(input_path)     #if a single file, just add filename to the image list
    else:
        sys.exit("Do not recognise "+input_path+" as directory or file")

    #If cropping, get list of roi files with matching image names
    if processing == "crop" or processing == "workflow_crop":
        if os.path.isdir(roi_file):
            for img in img_list:
                img_name = os.path.basename(os.path.splitext(img)[0])
                roi_temp = roi_file +os.sep+ img_name + ".zip" 

                if os.path.exists(roi_temp):
                    roi_list.append(roi_temp)
                else:
                    sys.exit("Cannot find ROI file for "+img)
                    
            print("List of ROIs: ", roi_list)
        elif os.path.isfile(roi_file):
            roi_list.append(roi_file)
        assert len(roi_list) == len(img_list), "Image and ROI lists do not match"
    else:
        #add list of empty strings so that it can run through for loop
        no_files = len(img_list)
        roi_list =[""]*no_files
      
    
    #loop through list of images and rois
    for img,roi_path in zip(img_list,roi_list):  
        print("Processing Image "+img)
        if processing == "crop" or processing == "workflow_crop":
            print("Processing ROI "+roi_path)
        aics_img = AICSImage(img)
        
        #check if scene valid; if not it iterates through all scenes
        len_scenes = len(aics_img.scenes)
        for scene in range(len_scenes):
            aics_img.set_scene(scene)
            test = aics_img.get_image_dask_data("YX",T=0,C=0,Z=0)
            try:
                test_max = test.max().compute()
                if test_max:
                    print(f"Scene {scene} is valid")
                    break
            except Exception as e:
                print(f"Scene {scene} not valid")
        
        lattice = LatticeData(aics_img,deskew_angle,skew_dir,dx,dy,dz,channel_dimension)

        if args.time_range:
            time_start,time_end = args.time_range
        else:
            time_start,time_end = 0,lattice.time
    
        if args.channel_range:
            channel_start, channel_end = args.channel_range
        else:
            channel_start, channel_end = 0,lattice.channels
        
        
        #implement deconvolution
        #implement deconvolution
        if args.deconvolution[0]:
            lattice.decon_processing = args.deconvolution[0].lower()
            #define the psf paths
            psf_ch1_path = ""
            psf_ch2_path = ""
            psf_ch3_path = ""
            psf_ch4_path = ""

            #assign psf paths to variables
            #if doesn't exist, skip
            try:
                psf_ch1_path = args.deconvolution_psf[0].replace(",","").strip()
                psf_ch2_path = args.deconvolution_psf[1].replace(",","").strip()
                psf_ch3_path = args.deconvolution_psf[2].replace(",","").strip()
                psf_ch4_path = args.deconvolution_psf[3].replace(",","").strip()
            except IndexError:
                pass
            
            #add a terminal flag for when calling commands that are used in gui
            lattice.psf = []
            lattice.otf_path =[]
            #set number of iterations
            if args.deconvolution_num_iter:
                lattice.psf_num_iter=args.deconvolution_num_iter
            else:
                lattice.psf_num_iter=10
                
            _read_psf(psf_ch1_path,
                psf_ch2_path,
                psf_ch3_path,
                psf_ch4_path,
                use_gpu_decon = lattice.decon_processing,
                LLSZWidget = None,
                lattice = lattice,
                terminal = True,
                )
            psf_arg = "psf"
            
        else:
            lattice.decon_processing = None

        #Override pixel values by reading metadata if file is czi
        if os.path.splitext(img)[1] == ".czi":
            dz,dy,dx = lattice.dz, lattice.dy, lattice.dx
            print(f"Pixel values from metadata (zyx): {dz},{dy},{dx}")
        
        #Setup workflows based on user input 
        if processing == "workflow" or processing == "workflow_crop":
            #load workflow from path
            workflow_path = Path(args.workflow_path[0])
            
            #load custom modules (*.py) in same directory as workflow file
            import importlib
            parent_dir = workflow_path.resolve().parents[0].__str__()+os.sep
            print(parent_dir)
            sys.path.append(parent_dir)
            custom_py_files = get_all_py_files(parent_dir)
            if len(custom_py_files)>0: 
                modules = map(importlib.import_module,custom_py_files)
                print(f"Custom modules imported {modules}") 
            
            #workflow has to be reloaded for each image and reinitialised
            user_workflow = load_workflow(workflow_path.__str__())
            assert type(user_workflow) is Workflow, "Workflow file is not a napari workflow object. Check file!"       
        
            input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(user_workflow)
            print(input_arg_first, input_arg_last, first_task_name,last_task_name)
            
            #get list of tasks
            task_list = list(user_workflow._tasks.keys())
            
            print("Workflow loaded:")
            print(user_workflow)
            
            task_name_start = first_task_name[0]
            try:
                task_name_last = last_task_name[0]
            except IndexError:
                task_name_last = task_name_start
            
            #if workflow involves cropping, assign first task as crop_volume_deskew 
            if processing == "workflow_crop":
               deskewed_shape = lattice.deskew_vol_shape
               deskewed_volume = da.zeros(deskewed_shape)
               z_start = 0
               z_end = deskewed_shape[0]
               roi = "roi"
               volume = "volume"
               #Create workflow for cropping and deskewing
               #volume and roi used will be set dynamically
               user_workflow.set("crop_deskew",crop_volume_deskew,
                                 original_volume = volume,
                                 deskewed_volume = deskewed_volume,
                                 roi_shape = roi,
                                 angle_in_degrees = deskew_angle,
                                 voxel_size_x = dx,
                                 voxel_size_y= dy,
                                 voxel_size_z = dz, 
                                 z_start = z_start, 
                                 z_end = z_end)
               #change the first task so it accepts "crop_deskew as input"
               new_task = modify_workflow_task(old_arg=input_arg_first,task_key=task_name_start,new_arg="crop_deskew",workflow=user_workflow)
               user_workflow.set(task_name_start,new_task) 

            elif processing == "workflow":
                #Verify if deskewing function is in workflow; if not, add as first task
                if user_workflow.get_task(task_name_start)[0] not in (cle.deskew_y,cle.deskew_x):
                    custom_workflow = True
                    input = "input"
                                #add task to the workflow
                    user_workflow.set("deskew_image",cle.deskew_y, 
                                                input_image =input,
                                                angle_in_degrees = deskew_angle,
                                                voxel_size_x = dx,
                                                voxel_size_y= dy,
                                                voxel_size_z = dz)
                                #Set input of the workflow to be from deskewing
                                #change the first task so it accepts "deskew_image" as input
                    new_task = modify_workflow_task(old_arg=input_arg_first,task_key=task_name_start,new_arg="deskew_image",workflow=user_workflow)
                    user_workflow.set(task_name_start,new_task)
                else:
                    custom_workflow = False
        
        img_data = lattice.data

        save_name = os.path.splitext(os.path.basename(img))[0]

        #Create save directory for each image
        save_path = output_path + os.sep + os.path.basename(os.path.splitext(img)[0]) + os.sep
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        print("Saving at ",save_path)

        if not args.output_file_type:
            output_file_type = 'tif'
        else:
            output_file_type = args.output_file_type[0]

        print(output_file_type)

        #Deskewing only
        if processing == "deskew": 
            
            #deconvolution
            if lattice.decon_processing:
                save_img(vol = img_data,
                        func = cle.deskew_y,
                        time_start = time_start,
                        time_end = time_end,
                        channel_start = channel_start,
                        channel_end = channel_end,
                        save_path = save_path,
                        save_name= save_name,
                        save_file_type = output_file_type,
                        dx = dx,
                        dy = dy,
                        dz = dz,
                        angle = deskew_angle,
                        terminal = True,
                        lattice = lattice,
                        angle_in_degrees = deskew_angle,
                        voxel_size_x=dx,
                        voxel_size_y=dy,
                        voxel_size_z=dz
                        )    
                
            else:
                save_img(vol = img_data,
                        func = cle.deskew_y,
                        time_start = time_start,
                        time_end = time_end,
                        channel_start = channel_start,
                        channel_end = channel_end,
                        save_path = save_path,
                        save_name= save_name,
                        save_file_type = output_file_type,
                        dx = dx,
                        dy = dy,
                        dz = dz,
                        angle = deskew_angle,
                        angle_in_degrees = deskew_angle,
                        voxel_size_x=dx,
                        voxel_size_y=dy,
                        voxel_size_z=dz
                        )
        
        #Crop and deskew
        elif processing == "crop" or processing =="workflow_crop":

            roi_img = read_imagej_roi(roi_path)
            
            for idx, roi_layer in enumerate(tqdm(roi_img, desc="ROI:", position=0)):
                print("Processing ROI "+str(idx)+" of "+str(len(roi_img)))
                deskewed_shape = lattice.deskew_vol_shape
                deskewed_volume = da.zeros(deskewed_shape)

                
                #Can modify for entering custom z values
                z_start = 0
                z_end = deskewed_shape[0]
                
                if processing == "crop":
                    #deconvolution
                    if lattice.decon_processing:
                        save_img(img_data,
                                    func = crop_volume_deskew,
                                    time_start = time_start,
                                    time_end = time_end,
                                    channel_start = channel_start,
                                    channel_end = channel_end,
                                    save_name_prefix  = "ROI_" + str(idx)+"_",
                                    save_path = save_path,
                                    save_name= save_name,
                                    save_file_type=output_file_type,
                                    dx = dx,
                                    dy = dy,
                                    dz = dz,
                                    angle = deskew_angle,
                                    terminal = True,
                                    lattice = lattice,
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
                        save_img(img_data,
                                    func = crop_volume_deskew,
                                    time_start = time_start,
                                    time_end = time_end,
                                    channel_start = channel_start,
                                    channel_end = channel_end,
                                    save_name_prefix  = "ROI_" + str(idx)+"_",
                                    save_path = save_path,
                                    save_name= save_name,
                                    save_file_type=output_file_type,
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
                    
                elif processing =="workflow_crop":
                    #deconvolution
                    user_workflow.set(roi,roi_layer)
                    
                    if lattice.decon_processing:
                        
                        save_img_workflow(vol=img_data,
                                       workflow = user_workflow,
                                       input_arg = volume,
                                       first_task = "crop_deskew",
                                       last_task = task_name_last,
                                       time_start = time_start,
                                       time_end = time_end,
                                       channel_start = channel_start,
                                       channel_end = channel_end,
                                       save_path = save_path,
                                       save_name_prefix = "ROI_"+str(idx),
                                       save_name =  save_name,
                                       save_file_type=output_file_type,
                                       dx = dx,
                                       dy = dy,
                                       dz = dz,
                                       angle = deskew_angle,
                                       deconvolution=True,
                                       decon_processing=lattice.decon_processing,
                                       psf=lattice.psf,
                                       psf_arg=psf_arg)
                    else:
                        save_img_workflow(vol=img_data,
                                       workflow = user_workflow,
                                       input_arg = volume,
                                       first_task = "crop_deskew",
                                       last_task = task_name_last,
                                       time_start = time_start,
                                       time_end = time_end,
                                       channel_start = channel_start,
                                       channel_end = channel_end,
                                       save_path = save_path,
                                       save_file_type=output_file_type,
                                       save_name_prefix = "ROI_"+str(idx),
                                       save_name =  save_name,
                                       dx = dx,
                                       dy = dy,
                                       dz = dz,
                                       angle = deskew_angle,
                                       deconvolution=False)
        
        elif processing == "workflow":
            #if deskew_image task set above manually
            if custom_workflow:
                if lattice.decon_processing:
                    save_img_workflow(vol=img_data,
                                   workflow = user_workflow,
                                   input_arg = input,
                                   first_task = "deskew_image",
                                   last_task = task_name_last,
                                   time_start = time_start,
                                   time_end = time_end,
                                   channel_start = channel_start,
                                   channel_end = channel_end,
                                   save_path = save_path,
                                   save_name =  save_name,
                                   save_file_type=output_file_type,
                                   dx = dx,
                                   dy = dy,
                                   dz = dz,
                                   angle = deskew_angle,
                                   deconvolution=True,
                                   decon_processing=lattice.decon_processing,
                                   psf=lattice.psf,
                                   psf_arg=psf_arg)
                else:
                    save_img_workflow(vol=img_data,
                                   workflow = user_workflow,
                                   input_arg = input,
                                   first_task = "deskew_image",
                                   last_task = task_name_last,
                                   time_start = time_start,
                                   time_end = time_end,
                                   channel_start = channel_start,
                                   channel_end = channel_end,
                                   save_path = save_path,
                                   save_name =  save_name,
                                   save_file_type=output_file_type,
                                   dx = dx,
                                   dy = dy,
                                   dz = dz,
                                   angle = deskew_angle)
            else:
                if lattice.decon_processing:
                    save_img_workflow(vol=img_data,
                                    workflow = user_workflow,
                                    input_arg = input_arg_first,
                                    first_task = first_task_name,
                                    last_task = task_name_last,
                                    time_start = time_start,
                                    time_end = time_end,
                                    channel_start = channel_start,
                                    channel_end = channel_end,
                                    save_path = save_path,
                                    save_name =  save_name,
                                    save_file_type=output_file_type,
                                    dx = dx,
                                    dy = dy,
                                    dz = dz,
                                    angle = deskew_angle,
                                    deconvolution=True,
                                    decon_processing=lattice.decon_processing,
                                    psf=lattice.psf,
                                    psf_arg=psf_arg)
                else:
                    save_img_workflow(vol=img_data,
                                    workflow = user_workflow,
                                    input_arg = input_arg_first,
                                    first_task = first_task_name,
                                    last_task = task_name_last,
                                    time_start = time_start,
                                    time_end = time_end,
                                    channel_start = channel_start,
                                    channel_end = channel_end,
                                    save_path = save_path,
                                    save_name =  save_name,
                                    save_file_type=output_file_type,
                                    dx = dx,
                                    dy = dy,
                                    dz = dz,
                                    angle = deskew_angle)
            

if __name__ == '__main__':
    main()