
# Channel-specific thresholding for a multi-channel workflow.
#
# napari-lattice runs the workflow once per channel and injects the current context as
# workflow inputs. We name `channel_index` as an argument to access the current channel
# (0, 1, ...) and choose a threshold per channel. This replaces the old
# `napari_lattice.config` module.

import numpy as np
from skimage.filters import threshold_triangle, threshold_otsu

def segment_multich(img, channel_index):
    # The incoming image may be a GPU (pyclesperanto) array; convert to numpy so the
    # scikit-image thresholds and the comparison below run on the host.
    img = np.asarray(img)
    #if first channel, use threshold_triangle
    if channel_index == 0:
        binary_img = img > threshold_triangle(img)
    #otherwise use Otsu threshold
    else:
        binary_img = img > threshold_otsu(img)
    return binary_img.astype(np.uint8)
