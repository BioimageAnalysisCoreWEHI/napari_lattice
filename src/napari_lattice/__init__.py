
__version__ = "0.0.1"


from enum import Enum


# Initialize configuration options
class DeskewDirection(Enum):
    X = 1
    Y = 2


class DeconvolutionChoice(Enum):
    cuda_gpu = 1
    opencl_gpu = 2
    cpu = 3
