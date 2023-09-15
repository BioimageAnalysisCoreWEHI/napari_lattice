from pydantic import Field, validator, DirectoryPath
from strenum import StrEnum
from os import getcwd
from pathlib import Path

from lls_core.models.utils import FieldAccessMixin, enum_choices

#Choice of File extension to save
class SaveFileType(StrEnum):
    h5 = "h5"
    tiff = "tiff"

class OutputParams(FieldAccessMixin, arbitrary_types_allowed=True):
    save_dir: DirectoryPath = Field(
        default = getcwd(),
        description="The directory where the output data will be saved"
    )
    save_name: str = Field(
        description="The filename prefix that will be used for output files, without a leading directory or file extension. The final output files will have additional elements added to the end of this prefix to indicate the region of interest, channel, timepoint, file extension etc."
    )
    save_type: SaveFileType = Field(
        default=SaveFileType.h5,
        description=f"The data type to save the result as. This will also be used to determine the file extension of the output files. Choices: {enum_choices(SaveFileType)}"
)
    time_range: range = Field(
        default = None,
        description="The range of times to process. This defaults to all time points in the image array."
    )
    channel_range: range = Field(
        description="The filename prefix that will be used for output files, without a leading directory or file extension. The final output files will have additional elements added to the end of this prefix to indicate the region of interest, channel, timepoint, file extension etc.",
        default=None
    )

    @property
    def file_extension(self):
        if self.save_type == SaveFileType.h5:
            return "h5"
        else:
            return "tif"

    def make_filepath(self, suffix: str) -> Path:
        """
        Returns a filepath for the resulting data
        """
        return self.save_dir / Path(self.save_name + suffix).with_suffix("." + self.file_extension)
