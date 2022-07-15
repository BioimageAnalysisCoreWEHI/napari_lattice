import numpy as np
from collections import defaultdict
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull, path
import os

import pandas as pd
import dask.array as da

import pyclesperanto_prototype as cle
from read_roi import read_roi_zip

from napari_workflows import Workflow
from tifffile import imsave


#get bounding box of ROI in 3D and the shape of the ROI
def calculate_crop_bbox(shape, z_start:int, z_end:int):
    """Get bounding box as vertices in 3D in the form xyz

    Args:
        shape (int): [description]
        z_start (int): Start of slice
        z_end (int): End of slice

    Returns:
        list,np.array: Bounding box of ROI in 3D (xyz), the shape of the ROI in 3D (zyx)
    """    
    #calculate bounding box and shape of cropped 3D volume
    #shape is 3D ROI from shape layer
    #get ROI coordinates,clip to zero, get shape of ROI volume and vertices of 3D ROI
    start = np.rint(np.min(shape, axis=0)).clip(0)
    stop = np.rint(np.max(shape, axis=0)).clip(0)

    #Shapes layer can return 3D coordinates, so only take last two
    if len(start>2): start = start[-2:]
    if len(stop>2): stop = stop[-2:]

    start=np.insert(start,0,z_start) # start[0]=z_start
    stop = np.insert(stop,0,z_end) #  =z_start

    z0,y0,x0 = np.stack([start,stop])[0].astype(int)
    z1,y1,x1 = np.stack([start,stop])[1].astype(int)

    crop_shape = (stop - start).astype(int).tolist()

    from itertools import product

    crop_bounding_box = [list(x)+[1] for x in product((x0,x1),(y0,y1),(z0,z1))] 
    return crop_bounding_box,crop_shape

def get_deskewed_shape(volume,
                        angle,
                        voxel_size_x_in_microns:float,
                        voxel_size_y_in_microns:float,
                        voxel_size_z_in_microns:float):
    """
    Calculate shape of deskewed volume

    Args:
        volume ([type]): Volume to deskew
        angle ([type]): Deskewing Angle
        voxel_size_x_in_microns ([type])
        voxel_size_y_in_microns ([type])
        voxel_size_z_in_microns ([type])

    Returns:
        tuple: Shape of deskewed volume in zyx
    """
    from pyclesperanto_prototype._tier8._affine_transform import _determine_translation_and_bounding_box

    deskew_transform = cle.AffineTransform3D()

    deskew_transform._deskew_y(angle_in_degrees=angle,
                                voxel_size_x=voxel_size_x_in_microns,
                                voxel_size_y=voxel_size_y_in_microns,
                                voxel_size_z=voxel_size_z_in_microns)
    
    #TODO:need better handling of aics dask array
    if len(volume.shape) == 5:
        volume = volume[0,0,...]

    new_shape, new_deskew_transform, _ = _determine_translation_and_bounding_box(volume, deskew_transform)
    return new_shape


#https://stackoverflow.com/a/10076823
#Credit: Antoine Pinsard
#Convert xml elementree to dict
def etree_to_dict(t):
    """Parse an XML file and convert to dictionary 
    This can be used to access the Zeiss metadata
    Access it from ["ImageDocument"]["Metadata"]

    Args:
        xml object : XML document (ImageDocument) containing Zeiss metadata

    Returns:
        dictionary: Zeiss czi file metadata 
    """    
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d

#block printing to console
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
        
#dask implementation for expand_dims not in latest release yet, so copying from their repo
#https://github.com/dask/dask/blob/dca10398146c6091a55c54db3778a06b485fc5ce/dask/array/routines.py#L1889        
def dask_expand_dims(a,axis):
    if type(axis) not in (tuple, list):
        axis = (axis,)

    out_ndim = len(axis) + a.ndim

    shape_it = iter(a.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]

    return a.reshape(shape)

def read_imagej_roi(roi_zip_path):
    """Read an ImageJ ROI zip file so it loaded into napari shapes layer
        If non rectangular ROI, will convert into a rectangle based on extreme points
    Args:
        roi_zip_path (zip file): ImageJ ROI zip file

    Returns:
        list: List of ROIs
    """    
    assert  path.splitext(roi_zip_path)[1] == ".zip", "ImageJ ROI file needs to be a zip file"
    ij_roi = read_roi_zip(roi_zip_path)

    #initialise list of rois
    roi_list = []

    #Read through each roi and create a list so that it matches the organisation of the shapes from napari shapes layer
    for k in ij_roi.keys():
        if ij_roi[k]['type'] in ('oval', 'rectangle'):
            width=ij_roi[k]['width']
            height=ij_roi[k]['height']
            left=ij_roi[k]['left']
            top=ij_roi[k]['top']
            roi = [[top,left],[top,left+width],[top+height,left+width],[top+height,left]]
            roi_list.append(roi)
        elif ij_roi[k]['type'] in ('polygon','freehand'):
            left = min(ij_roi[k]['x'])
            top = min(ij_roi[k]['y'])
            right = max(ij_roi[k]['x'])
            bottom = max(ij_roi[k]['y'])
            roi = [[top,left],[top,right],[bottom,right],[bottom,left]]
            roi_list.append(roi)
        else:
            print("Cannot read ROI ",ij_roi[k],".Recognised as type ",ij_roi[k]['type'])
    return roi_list

#Functions to deal with cle workflow
#TODO: Clean up this function
def get_first_last_image_and_task(user_workflow:Workflow):
    """Get images and tasks for first and last entry
    Args:
        user_workflow (Workflow): _description_
    Returns:
        list: name of first input image, last input image, first task, last task
    """    
    
    #get image with no preprocessing step (first image)
    input_arg_first = user_workflow.roots()[0]
    #get last image
    input_arg_last = user_workflow.leafs()[0]
    #get name of preceding image as that is the input to last task
    img_source = user_workflow.sources_of(input_arg_last)[0]
    first_task_name = []
    last_task_name = []

    #loop through workflow keys and get key that has 
    for key in user_workflow._tasks.keys():
        for task in user_workflow._tasks[key]:
            if task == input_arg_first:
                first_task_name.append(key)
            elif task == img_source:
                last_task_name.append(key)
                
    return input_arg_first, img_source, first_task_name, last_task_name


def modify_workflow_task(old_arg,task_key:str,new_arg,workflow):
    """_Modify items in a workflow task
    Workflow is not modified, only a new task with updated arg is returned
    Args:
        old_arg (_type_): The argument in the workflow that needs to be modified
        new_arg (_type_): New argument
        task_key (str): Name of the task within the workflow
        workflow (_type_): Workflow

    Returns:
        tuple: Modified task with name task_key
    """    
    task = workflow._tasks[task_key]
    #convert tuple to list for modification
    task_list = list(task)
    try:
        item_index =  task_list.index(old_arg)
    except ValueError:
       print(old_arg," not found in workflow file")
    task_list[item_index] = new_arg
    modified_task=tuple(task_list)
    return modified_task


def get_all_py_files(directory): 
    """get all py files within directory and return as a list of filenames
    Args:
        directory: Directory with .py files
    """    
    from os.path import dirname, basename, isfile, join
    import glob
    
    import sys
    
    modules = glob.glob(join(dirname(directory), "*.py"))
    all = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
    print(f"Files found are: {all}")
    
    
    return all

def as_type(img,ref_vol):
    """return image same dtype as ref_vol

    Args:
        img (_type_): _description_
        ref_vol (_type_): _description_

    Returns:
        _type_: _description_
    """    
    img.astype(ref_vol.dtype)
    return img

def process_custom_workflow_output(workflow_output,
                                 save_dir=None,
                                 idx=None,
                                 LLSZWidget=None,
                                 widget_class=None,
                                 channel=0,
                                 time=0,
                                 preview:bool=True):
    """Check the output from a custom workflow; 
    saves tables and images separately

    Args:
        workflow_output (_type_): _description_
        save_dir (_type_): _description_
        idx (_type_): _description_
        LLSZWidget (_type_): _description_
        widget_class (_type_): _description_
        channel (_type_): _description_
        time (_type_): _description_
    """    
    if type(workflow_output) in [dict,list]:
        #create function for tthis dataframe bit
        df = pd.DataFrame(workflow_output)
        if preview:
            save_path = path.join(save_dir,"lattice_measurement_"+str(idx)+".csv")
            print(f"Detected a dictionary as output, saving preview at",save_path)
            df.to_csv(save_path, index=False)
            return df
            
        else:
            return df
    elif type(workflow_output) in [np.ndarray,cle._tier0._pycl.OCLArray, da.core.Array]:
        if preview:
            suffix_name = str(idx)+"_c" + str(channel) + "_t" +str(time)
            scale = (LLSZWidget.LlszMenu.lattice.new_dz,LLSZWidget.LlszMenu.lattice.dy,LLSZWidget.LlszMenu.lattice.dx)
            widget_class.parent_viewer.add_image(workflow_output, name="Workflow_preview_"+ suffix_name,scale=scale)
        else:
            return workflow_output
        


def _process_custom_workflow_output_batch(ref_vol,
                                          no_elements,
                                          array_element_type,
                                          channel_range,
                                          images_array,
                                          save_path,
                                          time_point,
                                          ch,
                                          save_name_prefix,
                                          save_name,
                                          dx=None,
                                          dy=None,
                                          new_dz=None
                                          ):
    #create columns index for the list
    if list in array_element_type:
        row_idx=[]
        
     #Iterate through the dict or list output from workflow and add columns for Channel and timepoint
    for i in range(no_elements):
        for j in channel_range:
            if type(images_array[j,i]) in [dict]:
                #images_array[j,i].update({"Channel/Time":"C"+str(j)+"T"+str(time_point)})
                images_array[j,i].update({"Channel":"C"+str(j)})
                images_array[j,i].update({"Time":"T"+str(time_point)})
            elif type(images_array[j,i]) in [list]:
                row_idx.append("C"+str(j)+"T"+str(time_point))
                #row_idx.append("C"+str(j))
                #row_idx.append("T"+str(time_point))
    
    for element in range(no_elements):
        if(array_element_type[element]) in [dict]:
            #convert to pandas dataframe
            output_dict_pd = [pd.DataFrame(i) for i in images_array[:,element]]
            
            output_dict_pd = pd.concat(output_dict_pd)
            #set index to the channel/time
            output_dict_pd = output_dict_pd.set_index(["Time","Channel"])
            
            #Save path
            dict_save_path = os.path.join(save_path,"Measurement_"+save_name_prefix)
            if not(os.path.exists(dict_save_path)):
                os.mkdir(dict_save_path)
                
            #dict_save_path = os.path.join(dict_save_path,"C" + str(ch) + "T" + str(time_point)+"_"+str(element) + "_measurement.csv")
            dict_save_path = os.path.join(dict_save_path,"Summary_measurement_"+save_name_prefix+"_"+str(element)+"_.csv")
            #Opens csv and appends it if file already exists; not efficient.
            if os.path.exists(dict_save_path):
                output_dict_pd_existing = pd.read_csv(dict_save_path,index_col=["Time","Channel"])
                output_dict_summary = pd.concat((output_dict_pd_existing,output_dict_pd))
                output_dict_summary.to_csv(dict_save_path)
            else:
                output_dict_pd.to_csv(dict_save_path)
        
        #TODO:modify this so one file saved for measurement
        elif(array_element_type[element]) in [list]:
            
            output_list_pd = pd.DataFrame(np.vstack(images_array[:,element]),index=row_idx)
            #Save path
            list_save_path = os.path.join(save_path,"Measurement_"+save_name_prefix)
            if not(os.path.exists(list_save_path)):
                os.mkdir(list_save_path)
            list_save_path = os.path.join(list_save_path,"C" + str(ch) + "T" + str(time_point)+"_"+save_name_prefix+"_"+str(element) + "_measurement.csv")
            output_list_pd.to_csv(list_save_path)
        
        elif(array_element_type[element]) in [np.ndarray,cle._tier0._pycl.OCLArray, da.core.Array]:
            
            #Save path
            img_save_path = os.path.join(save_path,"Measurement_"+save_name_prefix)
            if not(os.path.exists(img_save_path)):
                os.mkdir(img_save_path)
            
            im_final = np.stack(images_array[:,element]).astype(ref_vol.dtype)
            final_name = os.path.join(img_save_path,save_name_prefix + "_"+str(element) + "_T" + str(
                time_point) + "_" + save_name  + ".tif") 
            #"C" + str(ch) + 
            #OmeTiffWriter.save(images_array, final_name, physical_pixel_sizes=aics_image_pixel_sizes)
            if len(im_final.shape) ==4: #if only one image with no channel, then dimension will 1,z,y,x, so swap 0 and 1
                im_final = np.swapaxes(im_final,0,1) #was 1,2,but when stacking images, dimension is CZYX
                im_final = im_final[np.newaxis,...] #adding extra dimension for T
            elif len(im_final.shape)>4:#if
                im_final = np.swapaxes(im_final,1,2) #if image with multiple channels, , it will be 1,c,z,y,x
            #imagej=True; ImageJ hyperstack axes must be in TZCYXS order
            imsave(final_name,im_final, bigtiff=True, imagej=True,resolution=(1./dx, 1./dy),
            metadata={'spacing': new_dz, 'unit': 'um', 'axes': 'TZCYX'})#imagej=True
            im_final = None
    return