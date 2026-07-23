#Sample code for multichannel prediction using cellpose in napari-workflows.
#
#napari-lattice runs the workflow once per channel and injects the current context as
#workflow inputs. A task can access them by naming them as arguments:
#   deskewed_image, channel, channel_index, time, time_index, roi_index
#Here we use `channel_index` (0 for the first channel, 1 for the second, ...) to pick a
#different cellpose model per channel. This replaces the old `napari_lattice.config` module.

import numpy as np
from cellpose import models


def predict_cellpose_multich(img, channel_index, model_channel1: str, model_channel2: str):
    #if first channel, use model specified in model_channel1
    if channel_index == 0:
        model_type = model_channel1
    #if second channel, use model specified in model_channel2
    elif channel_index == 1:
        model_type = model_channel2
    model = models.Cellpose(gpu=True, model_type=model_type)
    channels = [0, 0]
    img = np.array(img)
    masks, flows, styles, diams = model.eval(img, flow_threshold=None, channels=channels, diameter=25, do_3D=True)
    return masks
