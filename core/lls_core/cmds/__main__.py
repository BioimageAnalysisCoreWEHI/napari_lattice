# lattice_processing.py

# Run processing on command line instead of napari.
# Example for deskewing files in a folder
# python lattice_processing.py --input /home/pradeep/to_deskew --output /home/pradeep/output_save/ --processing deskew

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lls_core.models.lattice_data import (
    CropParams,
    LatticeData,
    OutputParams,
)
from lls_core.models.deskew import (
    DefinedPixelSizes,
    skew,
    angle,
    physical_pixel_sizes,
)
from lls_core.models.deconvolution import (
    Background,
    DeconvolutionParams,
    psf_num_iter,
    decon_processing,
    background
)
from lls_core.models.output import (
    SaveFileType,
    save_dir,
    save_type,
    channel_range,
    time_range
)
from napari_workflows import Workflow
from lls_core.models.crop import z_range
from pydantic_cli import run_and_exit, DefaultConfig, pydantic_class_to_parser
from pydantic import DirectoryPath, Field, NonNegativeInt, validator, FilePath, BaseModel, root_validator
from lls_core import DeconvolutionChoice, DeskewDirection

class 

class CliParams(BaseModel):
    # We can't directly use LatticeData because it uses nested models which aren't 
    # supported by pydantic-cli, and also we need to modify some types to support the CLI e.g. by allowing filepath inputs

    class Config(DefaultConfig):
        CLI_JSON_ENABLE = True
        arbitrary_types_allowed=True

    # Deskew
    image: FilePath = Field(description="Path to the image file to read, in a format readable by AICSImageIO, for example .tiff or .czi")
    skew: DeskewDirection = skew
    angle: float = angle
    physical_pixel_sizes: Optional[DefinedPixelSizes] = physical_pixel_sizes

    # Deconvolution
    deconvolution: bool = False
    decon_processing: DeconvolutionChoice = decon_processing
    psf: List[FilePath] = Field(default=[], description="A list of paths pointing to point spread functions to use for deconvolution. Each file should in a standard image format (.czi, .tiff etc), containing a 3D image array.")
    psf_num_iter = psf_num_iter
    background: Background = background

    # Crop
    roi_list: List[FilePath] = Field(default=[], description="A list of paths pointing to regions of interest to crop to, in ImageJ format.")
    z_range: Tuple[NonNegativeInt, NonNegativeInt] = z_range

    # Output
    # This one needs overriding to accommodate the default value
    save_name: str = Field(default=None, description=OutputParams.get_description("save_name"))
    save_dir: DirectoryPath = save_dir
    save_type: SaveFileType = save_type
    channel_range: range = channel_range
    time_range: range = time_range

    # Workflow
    workflow: Optional[FilePath] = Field(default = None, description="Path to a Napari Workflow file, in JSON format. If provided, the configured desekewing processing will be added to the chosen workflow.")

    @validator("save_name", pre=True)
    def default_save_name(cls, v: Any, values: Dict):
        # Use the input file path to provide a default output filename
        if v is None and values.get("image", None):
            path: Path = values["image"]
            return path.stem + "_deskewed"
        return v

    @root_validator(pre=True)
    def metadata_from_path(cls, values: Dict) -> Dict:

        return values

    def to_lattice(self) -> LatticeData:
        from aicsimageio import AICSImage

        crop = None
        if len(self.roi_list) > 0:
            crop = CropParams(
                roi_list=[AICSImage(it).data for it in self.roi_list],
                z_range=self.z_range
            )

        deconvolution = None
        if self.deconvolution:
            deconvolution = DeconvolutionParams(
                decon_processing=self.decon_processing,
                psf = [AICSImage(it).data for it in self.psf],
                psf_num_iter = self.psf_num_iter,
            )

        workflow = None
        if self.workflow is not None:
            from lls_core.workflow import workflow_from_path
            workflow = workflow_from_path(self.workflow)

        return LatticeData(
            image=AICSImage(self.image).xarray_dask_data,
            angle=self.angle,
            skew=self.skew,
            physical_pixel_sizes=self.physical_pixel_sizes,

            crop=crop,
            deconvolution=deconvolution,
            workflow=workflow,

            save_type=self.save_type,
            channel_range=self.channel_range,
            time_range=self.time_range,
            save_dir=self.save_dir,
            save_name=self.save_name
        )

def make_parser():
    parser = pydantic_class_to_parser(CliParams, default_value_override={})
    parser.add_argument("--yaml-config", required=False, help="Path to configuration YAML file")
    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    arg_dict = vars(args)

    json_path = arg_dict.pop("json-config", None)
    json_config = {}
    if json_path:
        import json
        with open(json_path) as fp:
            json_config = json.load(fp)

    yaml_path = arg_dict.pop("yaml-config", None)
    yaml_config = {}
    if yaml_path:
        import yaml
        with open(yaml_path) as fp:
            json_config = yaml.safe_load(fp)

    config = CliParams(
        **json_config,
        **yaml_config,
        **arg_dict
    )
    config.to_lattice().process().save_image()
