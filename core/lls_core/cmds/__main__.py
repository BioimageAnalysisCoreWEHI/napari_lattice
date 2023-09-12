# lattice_processing.py
# Run processing on command line instead of napari.
# Example for deskewing files in a folder
# python lattice_processing.py --input /home/pradeep/to_deskew --output /home/pradeep/output_save/ --processing deskew

from pathlib import Path
from typing import List

from lls_core.lattice_data import (
    CropParams,
    DeconvolutionParams,
    DeskewParams,
    LatticeData,
    OutputParams,
)
from pydantic_cli import run_and_exit, DefaultConfig
from pydantic import Field

class CliParams(DeskewParams, CropParams, OutputParams, DeconvolutionParams):
    # The idea of this class is to re-use the logic from the Pydantic
    # models, namely the validation, defaults etc, but to flatten them
    # into one model so that pydantic-cli can handle it, and to override
    # certain field definitions with CLI-specific data types
    save_name: str = Field(default=None, description=OutputParams.get_description("save_name"))
    image: Path = Field(description="Path to the image file to read, in a format readable by AICSImageIO, for example .tiff or .czi")
    roi_list: List[Path] = Field(default=[], description="A list of paths pointing to regions of interest to crop to, in ImageJ format.")
    psf: List[Path] = Field(default=[], description="A list of paths pointing to point spread functions to use for deconvolution. Each file should in a standard image format (.czi, .tiff etc), containing a 3D image array.")

    def to_lattice(self) -> LatticeData:
        return LatticeData(
            **self.dict()
        )

    class Config(DefaultConfig):
        CLI_JSON_ENABLE = True

def _main(params: CliParams) -> int:
    params.to_lattice().process().save_image()
    return 0

if __name__ == '__main__':
    run_and_exit(CliParams, _main)
