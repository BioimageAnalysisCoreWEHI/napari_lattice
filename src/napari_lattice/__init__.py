
__version__ = "0.0.1"


from enum import Enum

# Initialize configuration options

#Deskew Direction
class DeskewDirection(Enum):
    X = "X"
    Y = "Y"

#Choice of Deconvolution
class DeconvolutionChoice(Enum):
    cuda_gpu = "cuda_gpu"
    opencl_gpu = "opencl_gpu"
    cpu = "cpu"

#Choice of File extension to save
class SaveFileType(Enum):
    h5 = "h5"
    tiff = "tiff"
    