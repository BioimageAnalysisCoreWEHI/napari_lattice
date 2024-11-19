from typing import Any, List, Optional
import pytest
from lls_core.models import LatticeData
from lls_core.models.crop import CropParams
from lls_core.models.output import SaveFileType
from lls_core.sample import resources
from importlib_resources import as_file
import tempfile
from pathlib import Path
from napari_workflows import Workflow
from pytest import FixtureRequest


from .params import parameterized

root = Path(__file__).parent / "data"


def open_psf(name: str):
    with as_file(resources / "psfs" / "zeiss_simulated" / name) as path:
        return path


@parameterized
def test_process(minimal_image_path: str, args: dict):
    # Processes a minimal set of images, with multiple parameter combinations
    for slice in (
        LatticeData.parse_obj({"input_image": minimal_image_path, **args})
        .process()
        .slices
    ):
        assert slice.data.ndim == 3


def test_process_all(image_path: str):
    # Processes all input images, but without parameter combinations
    for slice in (
        LatticeData.parse_obj({"input_image": image_path}).process().slices
    ):
        assert slice.data.ndim == 3


@parameterized
def test_save(minimal_image_path: str, args: dict):
    with tempfile.TemporaryDirectory() as tempdir:
        LatticeData.parse_obj(
            {"input_image": minimal_image_path, "save_dir": tempdir, **args}
        ).process().save_image()
        results = list(Path(tempdir).iterdir())
        assert len(results) > 0


def test_process_deconv_crop():
    for slice in (
        LatticeData.parse_obj(
            {
                "input_image": root / "raw.tif",
                "deconvolution": {
                    "psf": [root / "psf.tif"],
                },
                "crop": CropParams(
                    roi_list=[[[0, 0], [0, 110], [95, 0], [95, 110]]]
                ),
            }
        )
        .process()
        .slices
    ):
        assert slice.data.ndim == 3


def test_process_time_range(multi_channel_time: Path):
    from lls_core.models.output import SaveFileType

    with tempfile.TemporaryDirectory() as outdir:
        LatticeData.parse_obj(
            {
                "input_image": multi_channel_time,
                # Channels 2 & 3
                "channel_range": range(1, 3),
                # Time point 2
                "time_range": range(1, 2),
                "save_dir": outdir,
                "save_type": SaveFileType.h5,
            }
        ).save()


@pytest.mark.parametrize(["background"], [(1,), ("auto",), ("second_last",)])
def test_process_deconvolution(background: Any):
    import numpy as np
    with tempfile.TemporaryDirectory() as outdir:
        for slice in (
            LatticeData.parse_obj(
                {
                    # Use random sample data here, since we're not testing the correctness of the deconvolution
                    # but rather that all the code paths are functional
                    "input_image": np.random.random_sample((128, 128, 64)),
                    "deconvolution": {
                        "psf": [np.random.random_sample((28, 28, 28))],
                        "background": background,
                    },
                    "save_dir": outdir
                }
            )
            .process()
            .slices
        ):
            assert slice.data.ndim == 3


@pytest.mark.parametrize(
    ["workflow_name"], [("image_workflow",), ("table_workflow",)]
)
def test_process_workflow(
    request: FixtureRequest, lls7_t1_ch1: Path, workflow_name: str
):
    from pandas import DataFrame

    workflow: Workflow = request.getfixturevalue(workflow_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        for output in (
            LatticeData.parse_obj(
                {
                    "input_image": lls7_t1_ch1,
                    "workflow": workflow,
                    "save_dir": tmpdir
                }
            )
            .process_workflow()
            .process()
        ):
            assert output.roi_index is None or isinstance(output.roi_index, int)
            assert isinstance(output.data, (Path, DataFrame))

def test_table_workflow(
    lls7_t1_ch1: Path, table_workflow: Workflow
):
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        results = set(LatticeData.parse_obj(
            {
                "input_image": lls7_t1_ch1,
                "workflow": table_workflow,
                "save_dir": tmpdir
            }
        ).process_workflow().save())
        # There should be one output for each element of the tuple
        assert {result.name for result in results} == {'LLS7_t1_ch1_deskewed_output_3.csv', 'LLS7_t1_ch1_deskewed.h5', 'LLS7_t1_ch1_deskewed_output_1.csv', 'LLS7_t1_ch1_deskewed_output_2.csv'}

@pytest.mark.parametrize(
    ["roi_subset"],
    [
        [None],
        [[0]],
        [[0, 1]],
    ],
)
@parameterized
def test_process_crop_roi_file(args: dict, roi_subset: Optional[List[int]]):
    # Test cropping with a roi zip file, selecting different subsets from that file
    with as_file(resources / "RBC_tiny.czi") as lattice_path:
        rois = root / "crop" / "two_rois.zip"
        slices = list(
            LatticeData.parse_obj(
                {
                    "input_image": lattice_path,
                    "crop": {"roi_list": [rois], "roi_subset": roi_subset},
                    **args,
                }
            )
            .process()
            .slices
        )
        # Check we made the correct number of slices
        assert len(slices) == len(roi_subset) if roi_subset is not None else 2
        for slice in slices:
            # Check correct dimensionality
            assert slice.data.ndim == 3


def test_process_crop_workflow(table_workflow: Workflow):
    import pandas as pd
    # Test cropping with a roi zip file, selecting different subsets from that file
    with as_file(
        resources / "RBC_tiny.czi"
    ) as lattice_path, tempfile.TemporaryDirectory() as outdir:
        results = list(LatticeData.parse_obj(
            {
                "input_image": lattice_path,
                "workflow": table_workflow,
                "save_dir": outdir,
                "save_type": SaveFileType.h5,
                "crop": {
                    "roi_list": [root / "crop" / "two_rois.zip"],
                },
            }
        ).process_workflow().save())
        # Two separate H5 files should be created in this scenario: one for each ROI
        # There should be one H5 for each ROI
        image_results = [path for path in results if path.suffix == ".h5"]
        assert len(image_results) == 2
        # There should be three CSVs for each ROI, one for each workflow result
        csv_results = [path for path in results if path.suffix == ".csv"]
        assert len(csv_results) == 2 * 3
        for csv in csv_results:
            # Test for CSV validity
            pd.read_csv(csv)


@pytest.mark.parametrize(
    ["roi"],
    [
        [[[(174.0, 24.0), (174.0, 88.0), (262.0, 88.0), (262.0, 24.0)]]],
        [[[(174.13, 24.2), (173.98, 87.87), (262.21, 88.3), (261.99, 23.79)]]],
    ],
)
@parameterized
def test_process_crop_roi_manual(args: dict, roi: List):
    # Test manually provided ROIs, both with integer and float values
    with as_file(resources / "RBC_tiny.czi") as lattice_path:
        for slice in (
            LatticeData.parse_obj(
                {
                    "input_image": lattice_path,
                    "crop": {"roi_list": roi},
                    **args,
                }
            )
            .process()
            .slices
        ):
            assert slice.data.ndim == 3
