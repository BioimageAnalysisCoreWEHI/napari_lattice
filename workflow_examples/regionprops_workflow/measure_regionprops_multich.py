
# Thresholding used as a segmentation step in the multi-channel workflow.
#
# Note: earlier versions of this example selected a different threshold per
# channel via `napari_lattice.config`, which no longer exists. Workflows now
# run once per channel and the function simply receives the image slice, so we
# apply a single channel-agnostic threshold here.

from skimage.filters import threshold_otsu

def segment_multich(img):
    binary_img = img > threshold_otsu(img)
    return binary_img