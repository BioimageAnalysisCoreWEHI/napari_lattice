import dask.array as da

def get_deskew_arr(img_raw,deskew_shape:tuple,vol_shape,time:float=0,channel:int=0,scene:int=0,skew_dir:str="Y"):
    """Return a dask array of the same dimensions as deskewed image and 

    Args:
        img_stack ([type]): [description]
        deskew_shape (tuple): [description]
        vol_shape (tuple): [description]
        time (float, optional): [description]. Defaults to 0.
        channel (int, optional): [description]. Defaults to 0.
        scene (int, optional): [description]. Defaults to 0.
        skew_dir (str, optional): [description]. Defaults to "Y".
    """ 
    print(type(img_raw))
    if(type(img_raw)) is da.Array:
        try:
            if len(img_raw.shape)==5:
                img_stack=img_raw[time,channel,:,:,:]
            elif len(img_raw.shape)==4:
                img_stack=img_raw[time,:,:,:]
        except:
            print("Image shape must be either 4 or 5, i.e., contain channel and/time). Got shape "+img_raw.shape)
    else:
        img_stack=img_raw.get_image_dask_data("ZYX",T=time,C=channel,S=scene)
    deskew_size=deskew_shape
    deskew_chunk_size=deskew_shape #tuple((nz,deskewed_y,nx))

    #create an empty dask array same size as the deskewed image with all the z-slices
    deskew_img=da.zeros(deskew_size,dtype=img_raw.dtype,chunks=deskew_chunk_size)

    if(skew_dir=="Y"):
        #insert the image into the dask array for zeiss lls
        deskew_img[:,:vol_shape[1],:]=img_stack
    elif (skew_dir=="X"):
        #insert the image into the dask array for home-built lls
        deskew_img[...,:vol_shape[2]]=img_stack
    
    return deskew_img

    