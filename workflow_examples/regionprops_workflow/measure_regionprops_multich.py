
#Channel specific  thresholding

from napari_lattice import config
from skimage.filters import threshold_triangle, threshold_otsu

def segment_multich(img):
    #if first channel, use threshold_triangle
    if config.channel == 0:
        binary_img = threshold_triangle(img)
    #if second channel, use Otsu threshold
    elif config.channel == 1:
        binary_img = threshold_otsu(img)
    return binary_img