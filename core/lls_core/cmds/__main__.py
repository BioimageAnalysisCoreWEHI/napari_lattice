# lattice_processing.py

# Run processing on command line instead of napari.
# Example for deskewing files in a folder
# python lattice_processing.py --input /home/pradeep/to_deskew --output /home/pradeep/output_save/ --processing deskew

from enum import auto
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import Annotated
from strenum import StrEnum

from lls_core.models.lattice_data import LatticeData
from lls_core.models.deskew import DefinedPixelSizes, DeskewParams
from lls_core.models.deconvolution import DeconvolutionParams
from lls_core.models.crop import CropParams
from lls_core.models.output import OutputParams
from lls_core import DeconvolutionChoice, DeskewDirection
from lls_core.types import image_like_to_image
import typer

from lls_core.models.output import SaveFileType
from toolz.dicttoolz import merge

# class 

# class CliParams(BaseModel):
#     # We can't directly use LatticeData because it uses nested models which aren't 
#     # supported by pydantic-cli, and also we need to modify some types to support the CLI e.g. by allowing filepath inputs

#     class Config(DefaultConfig):
#         CLI_JSON_ENABLE = True
#         arbitrary_types_allowed=True

#     # Deskew
#     image: FilePath = Field(description="Path to the image file to read, in a format readable by AICSImageIO, for example .tiff or .czi")
#     skew: DeskewDirection = skew
#     angle: float = angle
#     physical_pixel_sizes: Optional[DefinedPixelSizes] = physical_pixel_sizes

#     # Deconvolution
#     deconvolution: bool = False
#     decon_processing: DeconvolutionChoice = decon_processing
#     psf: List[FilePath] = Field(default=[], description="A list of paths pointing to point spread functions to use for deconvolution. Each file should in a standard image format (.czi, .tiff etc), containing a 3D image array.")
#     psf_num_iter = psf_num_iter
#     background: Background = background

#     # Crop
#     roi_list: List[FilePath] = Field(default=[], description="A list of paths pointing to regions of interest to crop to, in ImageJ format.")
#     z_range: Tuple[NonNegativeInt, NonNegativeInt] = z_range

#     # Output
#     # This one needs overriding to accommodate the default value
#     save_name: str = Field(default=None, description=OutputParams.get_description("save_name"))
#     save_dir: DirectoryPath = save_dir
#     save_type: SaveFileType = save_type
#     channel_range: range = channel_range
#     time_range: range = time_range

#     # Workflow
#     workflow: Optional[FilePath] = Field(default = None, description="Path to a Napari Workflow file, in JSON format. If provided, the configured desekewing processing will be added to the chosen workflow.")

#     @validator("save_name", pre=True)
#     def default_save_name(cls, v: Any, values: Dict):
#         # Use the input file path to provide a default output filename
#         if v is None and values.get("image", None):
#             path: Path = values["image"]
#             return path.stem + "_deskewed"
#         return v

#     @root_validator(pre=True)
#     def metadata_from_path(cls, values: Dict) -> Dict:

#         return values

#     def to_lattice(self) -> LatticeData:
#         from aicsimageio import AICSImage

#         crop = None
#         if len(self.roi_list) > 0:
#             crop = CropParams(
#                 roi_list=[AICSImage(it).data for it in self.roi_list],
#                 z_range=self.z_range
#             )

#         deconvolution = None
#         if self.deconvolution:
#             deconvolution = DeconvolutionParams(
#                 decon_processing=self.decon_processing,
#                 psf = [AICSImage(it).data for it in self.psf],
#                 psf_num_iter = self.psf_num_iter,
#             )

#         workflow = None
#         if self.workflow is not None:
#             from lls_core.workflow import workflow_from_path
#             workflow = workflow_from_path(self.workflow)

#         return LatticeData(
#             image=AICSImage(self.image).xarray_dask_data,
#             angle=self.angle,
#             skew=self.skew,
#             physical_pixel_sizes=self.physical_pixel_sizes,

#             crop=crop,
#             deconvolution=deconvolution,
#             workflow=workflow,

#             save_type=self.save_type,
#             channel_range=self.channel_range,
#             time_range=self.time_range,
#             save_dir=self.save_dir,
#             save_name=self.save_name
#         )

# def make_parser():
#     parser = pydantic_class_to_parser(CliParams, default_value_override={})
#     parser.add_argument("--yaml-config", required=False, help="Path to configuration YAML file")
#     return parser
class CliDeskewDirection(StrEnum):
    X = auto()
    Y = auto()

def main(
    image: Path = typer.Argument(help="Path to the image file to read, in a format readable by AICSImageIO, for example .tiff or .czi"),
    skew: CliDeskewDirection = typer.Option(
        default=DeskewParams.get_default("skew").name,
        help=DeskewParams.get_description("skew")
    ),# DeskewParams.make_typer_field("skew"),
    angle: float = DeskewParams.make_typer_field("angle") ,
    pixel_sizes: Tuple[float, float, float] = typer.Option(
    (
        LatticeData.get_default("physical_pixel_sizes").X,
        LatticeData.get_default("physical_pixel_sizes").Y,
        LatticeData.get_default("physical_pixel_sizes").Z,
    ), help=DeskewParams.get_description("physical_pixel_sizes") + ". This takes three arguments, corresponding to the X Y and Z pixel dimensions respectively"
    ),

    rois: List[Path] = typer.Option([], help="A list of paths pointing to regions of interest to crop to, in ImageJ format."),
    # Ideally this and other range values would be defined as Tuples, but these seem to be broken: https://github.com/tiangolo/typer/discussions/667
    z_start: Optional[int] = typer.Option(None, help="The index of the first Z slice to use. All prior Z slices will be discarded."),
    z_end: Optional[int] = typer.Option(None, help="The index of the last Z slice to use. The selected index and all subsequent Z slices will be discarded."),

    enable_deconvolution: Annotated[bool, typer.Option("--deconvolution/--disable-deconvolution")] = False,
    decon_processing: DeconvolutionChoice = DeconvolutionParams.make_typer_field("decon_processing"),
    psf: Annotated[List[Path], typer.Option(help="A list of paths pointing to point spread functions to use for deconvolution. Each file should in a standard image format (.czi, .tiff etc), containing a 3D image array.")] = [],
    psf_num_iter: int = DeconvolutionParams.make_typer_field("psf_num_iter"),
    background: str = DeconvolutionParams.make_typer_field("background"),

    time_start: Optional[int] = typer.Option(None, help="Index of the first time slice to use (inclusive)"),
    time_end: Optional[int] = typer.Option(None, help="Index of the first time slice to use (exclusive)"),

    channel_start: Optional[int] = typer.Option(None, help="Index of the first channel slice to use (inclusive)"),
    channel_end: Optional[int] = typer.Option(None, help="Index of the first channel slice to use (exclusive)"),
    
    save_dir: Path = OutputParams.make_typer_field("save_dir"),
    save_name: Optional[str] = OutputParams.make_typer_field("save_name"),
    save_type: SaveFileType = OutputParams.make_typer_field("save_type"),

    workflow: Optional[Path] = typer.Option(None, help="Path to a Napari Workflow file, in JSON format. If provided, the configured desekewing processing will be added to the chosen workflow."),
    json_config: Optional[Path] = typer.Option(None),
    yaml_config: Optional[Path] = typer.Option(None)
):
    cli_args = dict(
        image=image,
        angle=angle,
        skew=DeskewDirection[skew.name],
        physical_pixel_sizes=DefinedPixelSizes(
            X=pixel_sizes[0],
            Y=pixel_sizes[1],
            Z=pixel_sizes[2],
        ),
        crop=None if len(rois) == 0 else dict(
            roi_list=rois,
            z_range=(z_start, z_end)
        ),
        deconvolution=None if not enable_deconvolution else dict(
            decon_processing = decon_processing,
            psf = psf,
            psf_num_iter = psf_num_iter,
            background = background
        ),
        workflow=None,

        time_range=(time_start, time_end),
        channel_range=(channel_start, channel_end),
        save_dir=save_dir,
        save_name=save_name,
        save_type=save_type
    )

    json_args = {}
    if json_config is not None:
        import json
        with json_config.open() as fp:
            json_args = json.load(fp)

    yaml_args = {}
    if yaml_config is not None:
        with yaml_config.open() as fp:
            from yaml import safe_load
            yaml_args = safe_load(fp)

    return LatticeData.parse_obj(merge(yaml_args, json_args, cli_args))

    # img_data = image_like_to_image(image)

    # crop = None
    # if len(rois) > 0:
    #     crop = CropParams.from_img_metadata(
    #         img_data,
    #         roi_list=[image_like_to_image(it) for it in rois],
    #         z_range=z_range
    #     )

    # decon = None
    # if enable_deconvolution:
    #     decon = DeconvolutionParams(
    #         decon_processing=decon_processing,
    #         # background=
    #     )  

    LatticeData(
    )

if __name__ == '__main__':
    typer.run(main)
    # parser = make_parser()
    # args = parser.parse_args()
    # arg_dict = vars(args)

    # json_path = arg_dict.pop("json-config", None)
    # json_config = {}
    # if json_path:
    #     import json
    #     with open(json_path) as fp:
    #         json_config = json.load(fp)

    # yaml_path = arg_dict.pop("yaml-config", None)
    # yaml_config = {}
    # if yaml_path:
    #     import yaml
    #     with open(yaml_path) as fp:
    #         json_config = yaml.safe_load(fp)

    # config = CliParams(
    #     **json_config,
    #     **yaml_config,
    #     **arg_dict
    # )
    # config.to_lattice().process().save_image()
