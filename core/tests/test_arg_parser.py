from lls_core.cmds.__main__ import make_parser

def test_voxel_parsing():
    # Tests that we can parse voxel lists correctly
    parser = make_parser()
    args = parser.parse_args([
            "--input", "input",
            "--output", "output",
            "--processing", "deskew",
            "--output_file_type", "tiff",
            "--voxel_sizes", "1", "1", "1"
    ])
    assert args.voxel_sizes == [1.0, 1.0, 1.0]