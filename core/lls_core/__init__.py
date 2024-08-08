
__version__ = "0.2.6"

from strenum import StrEnum

#Choice of Deconvolution
class DeconvolutionChoice(StrEnum):
    cuda_gpu = "cuda_gpu"
    opencl_gpu = "opencl_gpu"
    cpu = "cpu"
