# lattice_processing.py

# Run processing on command line instead of napari.
# Example for deskewing files in a folder
# python lattice_processing.py --input /home/pradeep/to_deskew --output /home/pradeep/output_save/ --processing deskew
from __future__ import annotations

from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple
from strenum import StrEnum

from lls_core.models.lattice_data import LatticeData
from lls_core.models.deskew import DefinedPixelSizes, DeskewParams
from lls_core.models.deconvolution import DeconvolutionParams
from lls_core.models.output import OutputParams
from lls_core.models.crop import CropParams
from lls_core import DeconvolutionChoice, DeskewDirection
from typer import Typer, Argument, Option

from lls_core.models.output import SaveFileType
from toolz.dicttoolz import merge_with, valfilter

if TYPE_CHECKING:
    from lls_core.models.utils import FieldAccessMixin
    from typing import Type, Any

class CliDeskewDirection(StrEnum):
    X = auto()
    Y = auto()

app = Typer(add_completion=False, rich_markup_mode="rich")

def format_default(default: str):
    """
    Given a string, formats it like a default value in Typer
    """
    from typer.rich_utils import STYLE_OPTION_DEFAULT
    default_style = STYLE_OPTION_DEFAULT
    return f"[{default_style}]\\[default: {default}][/{default_style}]"


def field_from_model(model: Type[FieldAccessMixin], field_name: str, extra_description: str = "", description: Optional[str] = None) -> Any:
    """
    Generates a type Field from a Pydantic model field
    """
    field = model.__fields__[field_name]

    from enum import Enum
    default = field.get_default()
    if isinstance(default, Enum):
        default = default.name

    if description is None:
        description = f"{field.field_info.description} {extra_description}"

    return Option(
        # We make all defaults None so they can be removed
        default = None,
        show_default=False,
        help=f"{description}\n{format_default(default)}"
    )

def handle_merge(values: list):
    if len(values) > 0:
        raise ValueError(f"A parameter has been passed multiple times! Got: {', '.join(values)}")

@app.command("dump-schema")
def dump_schema():
    import json
    import sys
    json.dump(LatticeData.to_definition_dict(), fp=sys.stdout, indent=4)

@app.command("process")
def main(
    image: Path = Argument(None, help="Path to the image file to read, in a format readable by AICSImageIO, for example .tiff or .czi", show_default=False),
    skew: CliDeskewDirection = field_from_model(DeskewParams, "skew"),# DeskewParams.make_typer_field("skew"),
    angle: float = field_from_model(DeskewParams, "angle") ,
    pixel_sizes: Tuple[float, float, float] = field_from_model(DeskewParams, "physical_pixel_sizes", extra_description="This takes three arguments, corresponding to the X Y and Z pixel dimensions respectively"),
    rois: List[Path] = field_from_model(CropParams, "roi_list", description="A list of paths pointing to regions of interest to crop to, in ImageJ format."), #Option([], help="A list of paths pointing to regions of interest to crop to, in ImageJ format."),
    # Ideally this and other range values would be defined as Tuples, but these seem to be broken: https://github.com/tiangolo/typer/discussions/667
    z_start: Optional[int] = Option(None, help="The index of the first Z slice to use. All prior Z slices will be discarded. " + format_default("0"), show_default=False),
    z_end: Optional[int] = Option(None, help="The index of the last Z slice to use. The selected index and all subsequent Z slices will be discarded. Defaults to the last z index of the image.", show_default=False),

    enable_deconvolution: bool = Option(False, "--deconvolution/--disable-deconvolution"),
    decon_processing: DeconvolutionChoice = DeconvolutionParams.make_typer_field("decon_processing"),
    psf: List[Path] = field_from_model(DeconvolutionParams, "psf", description="A list of paths pointing to point spread functions to use for deconvolution. Each file should in a standard image format (.czi, .tiff etc), containing a 3D image array."),
    psf_num_iter: int = field_from_model(DeconvolutionParams, "psf_num_iter"),
    background: str = field_from_model(DeconvolutionParams, "background"),

    time_start: Optional[int] = Option(None, help="Index of the first time slice to use (inclusive). Defaults to the first time index of the image.", show_default=False),
    time_end: Optional[int] = Option(None, help="Index of the first time slice to use (exclusive). Defaults to the last time index of the image.", show_default=False),

    channel_start: Optional[int] = Option(None, help="Index of the first channel slice to use (inclusive). Defaults to the first channel index of the image.", show_default=False),
    channel_end: Optional[int] = Option(None, help="Index of the first channel slice to use (exclusive). Defaults to the last channel index of the image.", show_default=False),
    
    save_dir: Path = field_from_model(OutputParams, "save_dir"),
    save_name: Optional[str] = field_from_model(OutputParams, "save_name"),
    save_type: SaveFileType = field_from_model(OutputParams, "save_type"),

    workflow: Optional[Path] = Option(None, help="Path to a Napari Workflow file, in JSON format. If provided, the configured desekewing processing will be added to the chosen workflow.", show_default=False),
    json_config: Optional[Path] = Option(None, show_default=False, help="Path to a JSON file from which parameters will be read."),
    yaml_config: Optional[Path] = Option(None, show_default=False, help="Path to a YAML file from which parameters will be read.")
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


    LatticeData.parse_obj(
        # Merge all three sources of config: YAML, JSON and CLI
        merge_with(
            handle_merge,
            # Remove None values from all dictonaries, so that they merge appropriately
            *[valfilter(lambda x: x is not None, it) for it in [yaml_args, json_args, cli_args]]
        )
    ).process().save_image()

if __name__ == '__main__':
    app()
