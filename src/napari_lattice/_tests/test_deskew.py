#filename and function name should start with "test_" when using pytest
import pyclesperanto_prototype as cle 
import numpy as np 


def test_deskew():

    raw = np.zeros((5,5,5))
    raw[2,0,0] = 1
 
    deskewed = cle.deskew_y(raw,angle_in_degrees=60)

    assert deskewed.shape == (4,8,5)
    assert deskewed[3,2,0] == 1

    
