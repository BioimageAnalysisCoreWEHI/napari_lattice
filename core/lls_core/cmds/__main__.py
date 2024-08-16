# lattice_processing.py

# Run processing on command line instead of napari.
# Example for deskewing files in a folder
# python lattice_processing.py --input /home/pradeep/to_deskew --output /home/pradeep/output_save/ --processing deskew
from __future__ import annotations

from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from strenum import StrEnum

from lls_core.models.lattice_data import LatticeData
from lls_core.models.deskew import DeskewParams, DefinedPixelSizes
from lls_core.models.deconvolution import DeconvolutionParams
from lls_core.models.output import OutputParams
from lls_core.models.crop import CropParams
from lls_core import DeconvolutionChoice
from typer import Typer, Argument, Option, Context, Exit
from typer.main import get_command

from lls_core.models.output import SaveFileType
from pydantic import ValidationError

if TYPE_CHECKING:
    from lls_core.models.utils import FieldAccessModel
    from typing import Type, Any, Iterable
    from rich.table import Table

class CliDeskewDirection(StrEnum):
    X = auto()
    Y = auto()

CLI_PARAM_MAP = {
    "input_image": ["input_image"],
    "angle": ["angle"],
    "skew": ["skew"],
    "pixel_sizes": ["physical_pixel_sizes"],
    "rois": ["crop", "roi_list"],
    "roi_indices": ["crop", "roi_subset"],
    "z_start": ["crop", "z_range", 0],
    "z_end": ["crop", "z_range", 1],
    "decon_processing": ["deconvolution", "decon_processing"],
    "psf": ["deconvolution", "psf"],
    "psf_num_iter": ["deconvolution", "psf_num_iter"],
    "background": ["deconvolution", "background"],
    "workflow": ["workflow"],
    "time_start": ["time_range", 0],
    "time_end": ["time_range", 1],
    "channel_start": ["channel_range", 0],
    "channel_end": ["channel_range", 1],
    "save_dir": ["save_dir"],
    "save_name": ["save_name"],
    "save_type": ["save_type"],
}

app = Typer(add_completion=False, rich_markup_mode="rich", no_args_is_help=True)

def field_from_model(model: Type[FieldAccessModel], field_name: str, extra_description: str = "", description: Optional[str] = None, default: Optional[Any] = None, **kwargs) -> Any:
    """
    Generates a type Field from a Pydantic model field
    """
    field = model.__fields__[field_name]

    from enum import Enum
    if default is None:
        default = field.get_default()
    if isinstance(default, Enum):
        default = default.name

    if description is None:
        description = f"{field.field_info.description} {extra_description}"

    return Option(
        default = default,
        help=description,
        **kwargs
    )

def handle_merge(values: list):
    if len(values) > 1:
        raise ValueError(f"A parameter has been passed multiple times! Got: {', '.join(values)}")
    return values[0]

def rich_validation(e: ValidationError) -> Table:
    """
    Converts 
    """
    from rich.table import Table

    table = Table(title="Validation Errors")
    table.add_column("Parameter")
    # table.add_column("Command Line Argument")
    table.add_column("Error")

    for error in e.errors():
        table.add_row(
            str(error["loc"][0]),
            str(error["msg"]),
        )

    return table

def pairwise(iterable: Iterable) -> Iterable:
    """
    An implementation of the pairwise() function in Python 3.10+
    See: https://docs.python.org/3.12/library/itertools.html#itertools.pairwise
    """
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def update_nested_data(data: Union[dict, list], keys: list, new_value: Any):
    current = data

    for key, next_key in pairwise(keys):
        next = {} if isinstance(next_key, str) else []
        if isinstance(current, dict):
            current = current.setdefault(key, next)
        elif isinstance(current, list):
            if key >= len(current):
                current.insert(key, next)
            current = current[key]
        else:
            raise ValueError(f"Unknown data type {type(current)}. Cannot traverse.")

    last_key = keys[-1]
    if isinstance(current, dict):
        current[last_key] = new_value
    elif isinstance(current, list):
        current.insert(last_key, new_value)
    else:
        raise ValueError(f"Unknown data type {type(current)}. Cannot traverse.")

# Example usage:
@app.command()
def process(
    ctx: Context,
    input_image: Path = Argument(None, help="Path to the image file to read, in a format readable by AICSImageIO, for example .tiff or .czi", show_default=False),
    skew: CliDeskewDirection = field_from_model(DeskewParams, "skew"),# DeskewParams.make_typer_field("skew"),
    angle: float = field_from_model(DeskewParams, "angle") ,
    pixel_sizes: Tuple[float, float, float] = field_from_model(DeskewParams, "physical_pixel_sizes", extra_description="This takes three arguments, corresponding to the Z, Y and X pixel dimensions respectively", default=(
        DefinedPixelSizes.get_default("Z"),
        DefinedPixelSizes.get_default("Y"),
        DefinedPixelSizes.get_default("X")
    )),

    rois: List[Path] = field_from_model(CropParams, "roi_list", description="A list of paths pointing to regions of interest to crop to, in ImageJ format."), #Option([], help="A list of paths pointing to regions of interest to crop to, in ImageJ format."),
    roi_indices: List[int] = field_from_model(CropParams, "roi_subset"),
    # Ideally this and other range values would be defined as Tuples, but these seem to be broken: https://github.com/tiangolo/typer/discussions/667
    z_start: Optional[int] = Option(0, help="The index of the first Z slice to use. All prior Z slices will be discarded.", show_default=False),
    z_end: Optional[int] = Option(None, help="The index of the last Z slice to use. The selected index and all subsequent Z slices will be discarded. Defaults to the last z index of the image.", show_default=False),

    enable_deconvolution: bool = Option(False, "--deconvolution/--disable-deconvolution", rich_help_panel="Deconvolution"),
    decon_processing: DeconvolutionChoice = field_from_model(DeconvolutionParams, "decon_processing", rich_help_panel="Deconvolution"),
    psf: List[Path] = field_from_model(DeconvolutionParams, "psf", description="One or more paths pointing to point spread functions to use for deconvolution. Each file should in a standard image format (.czi, .tiff etc), containing a 3D image array. This option can be used multiple times to provide multiple PSF files.", rich_help_panel="Deconvolution"),
    psf_num_iter: int = field_from_model(DeconvolutionParams, "psf_num_iter", rich_help_panel="Deconvolution"),
    background: str = field_from_model(DeconvolutionParams, "background", rich_help_panel="Deconvolution"),

    time_start: Optional[int] = Option(0, help="Index of the first time slice to use (inclusive). Defaults to the first time index of the image.", rich_help_panel="Output"),
    time_end: Optional[int] = Option(None, help="Index of the first time slice to use (exclusive). Defaults to the last time index of the image.", show_default=False, rich_help_panel="Output"),

    channel_start: Optional[int] = Option(0, help="Index of the first channel slice to use (inclusive). Defaults to the first channel index of the image.", rich_help_panel="Output"),
    channel_end: Optional[int] = Option(None, help="Index of the first channel slice to use (exclusive). Defaults to the last channel index of the image.", show_default=False, rich_help_panel="Output"),
    
    save_dir: Path = field_from_model(OutputParams, "save_dir", rich_help_panel="Output"),
    save_name: Optional[str] = field_from_model(OutputParams, "save_name", rich_help_panel="Output"),
    save_type: SaveFileType = field_from_model(OutputParams, "save_type", rich_help_panel="Output"),

    workflow: Optional[Path] = Option(None, help="Path to a Napari Workflow file, in YAML format. If provided, the configured desekewing processing will be added to the chosen workflow.", show_default=False),
    json_config: Optional[Path] = Option(None, show_default=False, help="Path to a JSON file from which parameters will be read."),
    yaml_config: Optional[Path] = Option(None, show_default=False, help="Path to a YAML file from which parameters will be read."),

    show_schema: bool = Option(default=False, help="If provided, image processing will not be performed, and instead a JSON document outlining the JSON/YAML options will be printed to stdout. This can be used to assist with writing a config file for use with the --json-config and --yaml-config options.")
) -> None:
    from click.core import ParameterSource
    from rich.console import Console

    console = Console(stderr=True)

    if show_schema:
        import json
        import sys
        json.dump(
            LatticeData.to_definition_dict(),
            sys.stdout,
            indent=4
        )
        return

    # Just print help if the user didn't provide any arguments
    if all(src != ParameterSource.COMMANDLINE for src in ctx._parameter_source.values()):
        print(ctx.get_help())
        raise Exit()

    from toolz.dicttoolz import merge_with
    cli_args = {}
    for source, dest in CLI_PARAM_MAP.items():
        from click.core import ParameterSource
        if ctx.get_parameter_source(source) != ParameterSource.DEFAULT:
            update_nested_data(cli_args, dest, ctx.params[source])

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

    try:
        lattice = LatticeData.parse_obj(
            # Merge all three sources of config: YAML, JSON and CLI
            merge_with(
                handle_merge,
                [yaml_args, json_args, cli_args]
            )
        )
    except ValidationError as e:
        console.print(rich_validation(e))
        raise Exit(code=1)
        
    lattice.save()
    console.print(f"Processing successful. Results can be found in {lattice.save_dir.resolve()}")

# Used by the docs
click_app = get_command(app)

def main():
    app()

if __name__ == '__main__':
    main()
