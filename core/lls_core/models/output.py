from pydantic.v1 import Field, DirectoryPath, validator
from strenum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING
from pandas import DataFrame
from lls_core.models.utils import FieldAccessModel, enum_choices

if TYPE_CHECKING:
    pass

class SaveFileType(StrEnum):
    """
    Choice of File extension to save
    """
    h5 = "h5"
    tiff = "tiff"

class OutputParams(FieldAccessModel):
    save_dir: DirectoryPath = Field(
        description="The directory where the output data will be saved. This can be specified as a `str` or `Path`."
    )
    save_suffix: str = Field(
        default="_deskewed",
        description="The filename suffix that will be used for output files. This will be added as a suffix to the input file name if the input image was specified using a file name. If the input image was provided as an in-memory object, the `save_name` field should instead be specified.",
        cli_description="The filename suffix that will be used for output files. This will be added as a suffix to the input file name if the --save-name flag was not specified."
    )
    save_name: str = Field(
        description="The filename that will be used for output files. This should not contain a leading directory or file extension. The final output files will have additional elements added to the end of this prefix to indicate the region of interest, channel, timepoint, file extension etc.",
        default=None
    )
    save_type: SaveFileType = Field(
        default=SaveFileType.h5,
        description=f"The data type to save the result as. This will also be used to determine the file extension of the output files. Choices: `{enum_choices(SaveFileType)}`. Choices can alternatively be specifed as `str`, for example `'tiff'`."
    )
    time_range: range = Field(
        default=None,
        description="The range of times to process. This defaults to all time points in the image array.",
        cli_description="The range of times to process, as an array with two items: the first and last time index. This defaults to all time points in the image array."
    )
    channel_range: range = Field(
        default=None,
        description="The range of channels to process. This defaults to all time points in the image array.",
        cli_description="The range of channels to process, as an array with two items: the first and last channel index. This defaults to all channels in the image array."
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
        return self.get_unique_filepath(self.save_dir / Path(self.save_name + suffix).with_suffix("." + self.file_extension))
    
    def make_filepath_df(self, suffix: str, result: DataFrame) -> Path:
        """
        Returns a filepath for the non-image data
        """
        return self.get_unique_filepath(self.save_dir / Path(self.save_name + suffix).with_suffix(".csv"))
    
    def get_unique_filepath(self, path: Path) -> Path:
        """
        Returns a unique filepath by appending a number to the filename if the file already exists.
        """
        counter = 1
        while path.exists():
            path = path.with_name(f"{path.stem}_{counter}{path.suffix}")
            counter += 1
        return path
