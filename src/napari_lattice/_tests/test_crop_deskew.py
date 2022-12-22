import pyclesperanto_prototype as cle 
import numpy as np 

from napari_lattice.llsz_core import crop_volume_deskew


def test_crop_deskew():
    raw = np.zeros((5,5,5))
    raw[2,4,2] = 10
    deskew_angle = 60 

    deskewed = cle.deskew_y(raw,angle_in_degrees=deskew_angle).astype(raw.dtype)

    #print(np.argwhere(deskewed>0))

    #Crop deskewed volume 
    ref_crop_deskew_img = deskewed[0:4,3:5,0:5]

    #Similarly, generate an roi with coordinates (x1=0,y1=3,z1=0) to (x2=5,y2=5,z2=5)
    #Use this for cropping deskewed volume to get matching area
    roi = np.array(((3,0),(3,5),(5,5),(5,0)))
    z1 = 0
    z2 = 5

    cropped_deskew_img = crop_volume_deskew(original_volume = raw,
                                            deskewed_volume = deskewed,
                                            roi_shape = roi,
                                            angle_in_degrees = deskew_angle,
                                            z_start = z1, 
                                            z_end = z2,
                                            linear_interpolation=True).astype(raw.dtype)

    assert cropped_deskew_img[0, 1, 2] == ref_crop_deskew_img[0,1,2]
    assert cropped_deskew_img[0, 0, 2] == ref_crop_deskew_img[0,0,2]
    assert ref_crop_deskew_img.shape == cropped_deskew_img.shape