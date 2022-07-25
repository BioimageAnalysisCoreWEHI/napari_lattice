import pyclesperanto_prototype as cle 
import numpy as np 

from napari_lattice.llsz_core import crop_volume_deskew


def test_crop_deskew():
    raw = np.zeros((5,5,5))
    raw[2,3,3] = 1
    deskew_angle = 60 

    deskewed = cle.deskew_y(raw,angle_in_degrees=deskew_angle).astype(raw.dtype)


    #Crop area of interest
    ref_crop_deskew_img = deskewed[1:3,3:5,3:5]

    #roi generated from crop array (x,y)
    roi = np.array(((3,3),(3,5),(5,5),(5,3)))
    #z
    z1 = 1
    z2 = 3

    cropped_deskew_img = crop_volume_deskew(original_volume = raw,
                                            deskewed_volume = deskewed,
                                            roi_shape = roi,
                                            angle_in_degrees = deskew_angle,
                                            z_start = z1, 
                                            z_end = z2).astype(raw.dtype)
    #check if both crops have matching value at position 0,1,0
    assert ref_crop_deskew_img[0,1,0] ==1
    assert cropped_deskew_img[0,1,0] ==1
    #check if shape of the cropped volumes match
    assert ref_crop_deskew_img.shape == cropped_deskew_img.shape