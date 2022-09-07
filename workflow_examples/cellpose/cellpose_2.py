#Sample code for multichannel prediction using cellpose in napari
#This module is to be accessed from workflow using napari-workflow
#the config module in napari_lattice allows access to currently active time or channel

import numpy as np
from cellpose import models
from napari_lattice import config


def predict_cellpose_multich(img,model_channel1:str,model_channel2:str):
    #if first channel, use model specified in model_channel1
    if config.channel == 0:
        model_type = model_channel1
    #if second channel, use model specified in model_channel2
    elif config.channel == 1:
        model_type = model_channel2
    model = models.Cellpose(gpu=True, model_type=model_type)
    channels = [0,0]
    img =np.array(img)
    masks, flows, styles, diams = model.eval(img, flow_threshold=None, channels=channels, diameter=25, do_3D=True)
    return masks