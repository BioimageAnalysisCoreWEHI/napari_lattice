from typer.testing import CliRunner
from typer.main import get_command
from lls_core.cmds.__main__ import app

runner = CliRunner()

def test_voxel_parsing():
    # Tests that we can parse voxel lists correctly
    parser = get_command(app).make_parser()
    parser.make_context()
    args = parser.parse_args(args=[
            "--input", "input",
            "--output", "output",
            "--processing", "deskew",
            "--output_file_type", "tiff",
            "--voxel_sizes", "1", "1", "1"
    ])
    assert args.voxel_sizes == [1.0, 1.0, 1.0]
