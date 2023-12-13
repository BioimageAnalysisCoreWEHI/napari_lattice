from pydantic import Field, DirectoryPath, validator
from strenum import StrEnum
from os import getcwd
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lls_core.models.utils import FieldAccessModel, enum_choices

if TYPE_CHECKING:
    pass

#Choice of File extension to save
class SaveFileType(StrEnum):
    h5 = "h5"
    tiff = "tiff"

class OutputParams(FieldAccessModel):
    save_dir: DirectoryPath = Field(
        description="The directory where the output data will be saved"
    )
    save_suffix: str = Field(
        default="_deskewed",
        description="The filename suffix that will be used for output files. This will be added as a suffix to the input file name if the input image was specified using a file name. If the input image was provided as an in-memory object, the `save_name` field should instead be specified."
    )
    save_name: str = Field(
        description="The filename that will be used for output files. This should not contain a leading directory or file extension. The final output files will have additional elements added to the end of this prefix to indicate the region of interest, channel, timepoint, file extension etc.",
        default=None
    )
    save_type: SaveFileType = Field(
        default=SaveFileType.h5,
        description=f"The data type to save the result as. This will also be used to determine the file extension of the output files. Choices: {enum_choices(SaveFileType)}"
    )
    time_range: range = Field(
        default=None,
        description="The range of times to process. This defaults to all time points in the image array."
    )
    channel_range: range = Field(
        description="The filename prefix that will be used for output files, without a leading directory or file extension. The final output files will have additional elements added to the end of this prefix to indicate the region of interest, channel, timepoint, file extension etc.",
        default=None
    )

    @validator("save_dir", pre=True)
    def validate_save_dir(cls, v: Path):
        if isinstance(v, Path) and not v.is_absolute():
            # This stops the empty path being considered a valid directory
            raise ValueError("The save directory must be an absolute path that exists")
        return v

    @validator("save_name")
    def add_save_suffix(cls, v: str, values: dict):
        # This is the only place that the save suffix is used.
        return v + values["save_suffix"]

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
