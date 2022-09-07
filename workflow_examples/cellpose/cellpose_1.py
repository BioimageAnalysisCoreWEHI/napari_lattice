import numpy as np
from cellpose import models


def predict_cellpose(img,model_type:str="cyto"):
    model = models.Cellpose(gpu=True, model_type=model_type)
    channels = [0,0]
    img =np.array(img)
    masks, flows, styles, diams = model.eval(img, flow_threshold=None, channels=channels, diameter=25, do_3D=True)
    return masks