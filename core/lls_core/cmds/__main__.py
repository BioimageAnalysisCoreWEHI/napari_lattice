# lattice_processing.py

# Run processing on command line instead of napari.
# Example for deskewing files in a folder
# python lattice_processing.py --input /home/pradeep/to_deskew --output /home/pradeep/output_save/ --processing deskew

from enum import auto
from pathlib import Path
from typing import List, Optional, Tuple
from typing_extensions import Annotated
from strenum import StrEnum

from lls_core.models.lattice_data import LatticeData
from lls_core.models.deskew import DefinedPixelSizes, DeskewParams
from lls_core.models.deconvolution import DeconvolutionParams
from lls_core.models.output import OutputParams
from lls_core import DeconvolutionChoice, DeskewDirection
from typer import Typer, Argument, Option

from lls_core.models.output import SaveFileType
from toolz.dicttoolz import merge

class CliDeskewDirection(StrEnum):
    X = auto()
    Y = auto()

app = Typer(add_completion=False)

@app.command()
def main(
    image: Path = Argument(help="Path to the image file to read, in a format readable by AICSImageIO, for example .tiff or .czi"),
    skew: CliDeskewDirection = Option(
        default=DeskewParams.get_default("skew").name,
        help=DeskewParams.get_description("skew")
    ),# DeskewParams.make_typer_field("skew"),
    angle: float = DeskewParams.make_typer_field("angle") ,
    pixel_sizes: Tuple[float, float, float] = Option(
    (
        LatticeData.get_default("physical_pixel_sizes").X,
        LatticeData.get_default("physical_pixel_sizes").Y,
        LatticeData.get_default("physical_pixel_sizes").Z,
    ), help=DeskewParams.get_description("physical_pixel_sizes") + ". This takes three arguments, corresponding to the X Y and Z pixel dimensions respectively"
    ),

    rois: List[Path] = Option([], help="A list of paths pointing to regions of interest to crop to, in ImageJ format."),
    # Ideally this and other range values would be defined as Tuples, but these seem to be broken: https://github.com/tiangolo/typer/discussions/667
    z_start: Optional[int] = Option(None, help="The index of the first Z slice to use. All prior Z slices will be discarded."),
    z_end: Optional[int] = Option(None, help="The index of the last Z slice to use. The selected index and all subsequent Z slices will be discarded."),

    enable_deconvolution: Annotated[bool, Option("--deconvolution/--disable-deconvolution")] = False,
    decon_processing: DeconvolutionChoice = DeconvolutionParams.make_typer_field("decon_processing"),
    psf: Annotated[List[Path], Option(help="A list of paths pointing to point spread functions to use for deconvolution. Each file should in a standard image format (.czi, .tiff etc), containing a 3D image array.")] = [],
    psf_num_iter: int = DeconvolutionParams.make_typer_field("psf_num_iter"),
    background: str = DeconvolutionParams.make_typer_field("background"),

    time_start: Optional[int] = Option(None, help="Index of the first time slice to use (inclusive)"),
    time_end: Optional[int] = Option(None, help="Index of the first time slice to use (exclusive)"),

    channel_start: Optional[int] = Option(None, help="Index of the first channel slice to use (inclusive)"),
    channel_end: Optional[int] = Option(None, help="Index of the first channel slice to use (exclusive)"),
    
    save_dir: Path = OutputParams.make_typer_field("save_dir"),
    save_name: Optional[str] = OutputParams.make_typer_field("save_name"),
    save_type: SaveFileType = OutputParams.make_typer_field("save_type"),

    workflow: Optional[Path] = Option(None, help="Path to a Napari Workflow file, in JSON format. If provided, the configured desekewing processing will be added to the chosen workflow."),
    json_config: Optional[Path] = Option(None),
    yaml_config: Optional[Path] = Option(None)
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
        workflow=workflow,

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

if __name__ == '__main__':
    app()
