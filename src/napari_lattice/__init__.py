
__version__ = "0.0.1"


from enum import Enum

# Initialize configuration options

#Deskew Direction
from pyclesperanto_prototype._tier8._affine_transform_deskew_3d import DeskewDirection


#Choice of Deconvolution
class DeconvolutionChoice(Enum):
    cuda_gpu = "cuda_gpu"
    opencl_gpu = "opencl_gpu"
    cpu = "cpu"

#Choice of File extension to save
class SaveFileType(Enum):
    h5 = "h5"
    tiff = "tiff"

#CONFIGURE LOGGING using a dictionary (can also be done with yaml file)
import logging.config
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'default_formatter': {
            'format': '[%(levelname)s:%(asctime)s] %(message)s'
        },
    },

    'handlers': {
        'stream_handler': {
            'class': 'logging.StreamHandler',
            'formatter': 'default_formatter',
        },
    },

    'loggers': {
        '': {
            'handlers': ['stream_handler'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

#Configuring logging level with empty string "" under the key loggers means its for root
#This will override levels for all other python libraries as they haven't been imported yet
#Specifying levels in each modules will override the root level

# Specify during initialization:
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
logger.debug("Logging is configured.")

#specify an enum for log levels
class Log_Levels(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    