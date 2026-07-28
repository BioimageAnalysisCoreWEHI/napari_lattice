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
from lls_core.cropping import RoiUnits
from lls_core.deconvolution import DeconvolutionChoice
from typer import Typer, Argument, Option, Context, Exit, BadParameter
from typer.main import get_command
import click

from lls_core.models.output import SaveFileType, MipInterpolation
from pydantic.v1 import ValidationError
from toolz.dicttoolz import merge_with

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
    "invert_scan_direction": ["invert_scan_direction"],
    "coverslip_rotation": ["coverslip_rotation"],
    "physical_pixel_sizes": ["physical_pixel_sizes"],
    "roi_list": ["crop", "roi_list"],
    "roi_units": ["crop", "roi_units"],
    "roi_subset": ["crop", "roi_subset"],
    "z_range": ["crop", "z_range"],
    "decon_processing": ["deconvolution", "decon_processing"],
    "psf": ["deconvolution", "psf"],
    "decon_num_iter": ["deconvolution", "decon_num_iter"],
    "background": ["deconvolution", "background"],
    "workflow": ["workflow"],
    "time_range": ["time_range"],
    "channel_range": ["channel_range"],
    "save_dir": ["save_dir"],
    "save_name": ["save_name"],
    "save_type": ["save_type"],
    "process_parallel": ["process_parallel"],
    "memory_safety_factor": ["memory_safety_factor"],
    "save_mip": ["save_mip"],
    "mip_interpolation": ["mip_interpolation"],
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

def parse_roi_subset(value: Optional[List[str]]) -> Optional[List[int]]:
    """
    Typer callback  normalises ``--roi-subset`` into a flat list of integer
    indices. Accepts repeated flags (``--roi-subset 2 --roi-subset 5``) and/or a
    comma-separated list (``--roi-subset 2,5,7``), with surrounding whitespace
    tolerated. A non-integer piece fails fast as a CLI usage error.
    """
    if not value:
        return value
    result: List[int] = []
    for item in value:
        for piece in str(item).split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                result.append(int(piece))
            except ValueError:
                raise BadParameter(f"ROI subset indices must be integers; got {piece!r}")
    return result

def handle_merge(values: list):
    if len(values) > 1:
        if all(isinstance(param, dict) for param in values):
            return merge_with(handle_merge, values)
        else:
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
@app.command(epilog="Run [bold]lls-pipeline estimate[/bold] with these same options to print a VRAM/RAM memory estimate and exit without processing (useful for sizing SLURM jobs or picking a value for [bold]--process-parallel[/bold]).")
def process(
    ctx: Context,
    input_image: Path = Argument(None, help="Path to the image file to read, in a format readable by AICSImageIO, for example .tiff or .czi", show_default=False),
    skew: CliDeskewDirection = field_from_model(DeskewParams, "skew"),# DeskewParams.make_typer_field("skew"),
    angle: float = field_from_model(DeskewParams, "angle") ,
    invert_scan_direction: bool = field_from_model(DeskewParams, "invert_scan_direction"),
    coverslip_rotation: bool = field_from_model(DeskewParams, "coverslip_rotation"),
    physical_pixel_sizes: Tuple[float, float, float] = field_from_model(DeskewParams, "physical_pixel_sizes", extra_description="This takes three arguments, corresponding to the Z, Y and X pixel dimensions respectively", default=(
        DefinedPixelSizes.get_default("Z"),
        DefinedPixelSizes.get_default("Y"),
        DefinedPixelSizes.get_default("X")
    )),

    roi_list: List[Path] = field_from_model(CropParams, "roi_list"),
    roi_units: RoiUnits = field_from_model(CropParams, "roi_units"),
    roi_subset: List[str] = field_from_model(CropParams, "roi_subset", extra_description="Accepts either repeated flags (--roi-subset 2 --roi-subset 5) or a comma-separated list (--roi-subset 2,5,7).", default=[], callback=parse_roi_subset),
    z_range: Optional[Tuple[int,int]] = field_from_model(CropParams, "z_range", show_default=False),
    
    enable_deconvolution: bool = Option(False, "--deconvolution/--disable-deconvolution", rich_help_panel="Deconvolution"),
    decon_processing: DeconvolutionChoice = field_from_model(DeconvolutionParams, "decon_processing", rich_help_panel="Deconvolution"),
    psf: List[Path] = field_from_model(DeconvolutionParams, "psf", rich_help_panel="Deconvolution"),
    decon_num_iter: int = field_from_model(DeconvolutionParams, "decon_num_iter", rich_help_panel="Deconvolution"),
    background: str = field_from_model(DeconvolutionParams, "background", rich_help_panel="Deconvolution"),

    time_range: Optional[Tuple[int,int]] = field_from_model(OutputParams, "time_range", rich_help_panel="Output"),
    channel_range: Optional[Tuple[int,int]] = field_from_model(OutputParams,"channel_range", rich_help_panel="Output"),
        
    save_dir: Path = field_from_model(OutputParams, "save_dir", rich_help_panel="Output"),
    save_name: Optional[str] = field_from_model(OutputParams, "save_name", rich_help_panel="Output"),
    save_type: SaveFileType = field_from_model(OutputParams, "save_type", rich_help_panel="Output"),
    process_parallel: int = field_from_model(OutputParams, "process_parallel", rich_help_panel="Output"),
    memory_safety_factor: float = field_from_model(OutputParams, "memory_safety_factor", rich_help_panel="Output"),
    save_mip: bool = field_from_model(OutputParams, "save_mip", rich_help_panel="Output"),
    mip_interpolation: MipInterpolation = field_from_model(OutputParams, "mip_interpolation", rich_help_panel="Output"),

    workflow: Optional[Path] = field_from_model(LatticeData, "workflow", show_default=False),
    json_config: Optional[Path] = Option(None, show_default=False, help="Path to a JSON file from which parameters will be read."),
    yaml_config: Optional[Path] = Option(None, show_default=False, help="Path to a YAML file from which parameters will be read."),

    show_schema: bool = Option(default=False, help="If provided, image processing will not be performed, and instead a JSON document outlining the JSON/YAML options will be printed to stdout. This can be used to assist with writing a config file for use with the --json-config and --yaml-config options.")
) -> None:
    from click.core import ParameterSource
    from rich.console import Console
    import logging

    console = Console(stderr=True)

    # Surface lls_core INFO logs (memory estimate, per-chunk failure
    # summaries) without turning on noisy INFO logging for every dependency.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("lls_core").setLevel(logging.INFO)

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

    # Merge all three sources of config: YAML, JSON and CLI
    merged = merge_with(handle_merge, [yaml_args, json_args, cli_args])
    # On the CLI, default to 'auto' (0) worker selection when the user set no value
    # anywhere. The model default stays 1 (serial) so GUI/library callers are
    # unaffected; only an unconfigured CLI run opts into auto.
    merged.setdefault("process_parallel", 0)

    try:
        lattice = LatticeData.parse_obj(merged)
    except ValidationError as e:
        console.print(rich_validation(e))
        raise Exit(code=1)

    # `estimate` and `process` are the same underlying command registered under
    # two names (see build_cli); the invoked subcommand name selects the mode.
    # `lls estimate ...` prints a VRAM/RAM memory estimate for the configured
    # pipeline and exits without processing (useful for sizing SLURM jobs or
    # picking a value for --process-parallel).
    if ctx.info_name == "estimate":
        from lls_core.estimate import estimate_pipeline
        try:
            report = estimate_pipeline(
                lattice,
                n_workers=lattice.process_parallel,
                safety_factor=lattice.memory_safety_factor,
            )
        except Exception as e:
            console.print(f"[red]Could not produce memory estimate:[/red] {e}")
            raise Exit(code=1)
        console.print(report.format_report())
        return

    try:
        lattice.save()
    except Exception as e:
        console.print(f"[red]Processing failed:[/red] {e}")
        raise Exit(code=1)
    console.print(f"Processing successful. Results can be found in {lattice.save_dir.resolve()}")

class DefaultCommandGroup(click.Group):
    """
    A Click group that falls back to a default subcommand when the first CLI
    token is not a recognised command. This keeps the pre-subcommand invocation
    style working after ``process``/``estimate`` were split into subcommands:
    ``lls <image> <options>`` and ``lls <options> <image>`` still route to
    ``process``, while ``lls estimate ...`` opts into the estimate path.

    The injection happens in ``parse_args`` (before group-level option parsing)
    so a *leading option* (e.g. ``lls --angle 30 img.tif``) is also routed to
    the default command rather than being rejected as an unknown group option.

    ``--help`` / bare invocation are routed to the default command too, so
    ``lls --help`` shows the full ``process`` option list rather than the sparse
    group help (which would list only the two subcommand names). ``estimate``
    stays discoverable via the ``process`` command's epilog.
    """
    default_cmd_name = "process"

    def parse_args(self, ctx, args):
        if not (args and args[0] in self.commands):
            args = [self.default_cmd_name, *args]
        return super().parse_args(ctx, args)

def build_cli() -> click.Group:
    """
    Assemble CLI as a group exposing ``process`` and ``estimate``.

    ``estimate`` reuses the exact same Click command object as ``process`` (same
    full option signature, no duplication); the two are distinguished at runtime
    via ``ctx.info_name`` inside :func:`process`. ``process`` is the default
    subcommand for backwards compatibility.
    """
    process_cmd = get_command(app)
    # A single-command Typer app returns the command itself; a multi-command app
    # returns a Group. Normalise to the bare `process` command either way.
    if isinstance(process_cmd, click.Group):
        process_cmd = process_cmd.commands["process"]
    group = DefaultCommandGroup(name="lls_core", no_args_is_help=True)
    group.add_command(process_cmd, name="process")
    group.add_command(process_cmd, name="estimate")
    return group

# Used by the docs and the console-script entry point
click_app = build_cli()

def main():
    click_app()

if __name__ == '__main__':
    main()
