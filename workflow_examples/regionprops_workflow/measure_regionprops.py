from skimage.measure import regionprops_table
import numpy as np

def measure_region_properties(label_img):
    label_np = np.array(label_img)
    measurements = regionprops_table(label_np, properties=( 'area',
                                                 'centroid',
                                                 'axis_major_length',
                                                 'axis_minor_length'))
    return measurements,label_np