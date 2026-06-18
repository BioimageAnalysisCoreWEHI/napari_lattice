"""
Tests for the parallel-ROI processing and pre-flight memory estimator.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from xarray import DataArray

from lls_core.estimate import (
    chunk_roi_subset,
    estimate_pipeline,
    estimate_roi,
    get_global_mem_size,
    get_max_allocation_size,
)
from lls_core.models.crop import CropParams
from lls_core.models.lattice_data import LatticeData


def _make_lattice(
    raw: np.ndarray,
    rois: List[List[List[int]]],
    tmpdir: str,
    process_parallel: int = 1,
) -> LatticeData:
    return LatticeData(
        input_image=DataArray(raw, dims=["Z", "Y", "X"]),
        physical_pixel_sizes=(1, 1, 1),
        save_name="test",
        save_dir=tmpdir,
        save_type="tiff",
        crop=CropParams(roi_list=rois, z_range=(0, 20)),
        process_parallel=process_parallel,
    )


# --- chunking ---------------------------------------------------------------

def test_chunk_roi_subset_even_split():
    assert chunk_roi_subset([0, 1, 2, 3], 2) == [[0, 2], [1, 3]]


def test_chunk_roi_subset_uneven_split():
    assert chunk_roi_subset([0, 1, 2, 3, 4], 3) == [[0, 3], [1, 4], [2]]


def test_chunk_roi_subset_fewer_rois_than_workers():
    # Fewer ROIs than requested workers should not create empty chunks
    assert chunk_roi_subset([0, 1], 4) == [[0], [1]]


def test_chunk_roi_subset_single_roi():
    assert chunk_roi_subset([7], 3) == [[7]]


# --- estimator pure shape math ---------------------------------------------

def test_estimate_roi_returns_nonzero_bboxes():
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    roi = [[[0, 0], [0, 40], [40, 40], [40, 0]]]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, roi, tmpdir)
        est = estimate_roi(lattice, 0)

    assert est.roi_index == 0
    # The deskew z range (0, 20) is the output Z depth
    assert est.output_crop_zyx[0] == 20
    # All three bboxes are non-trivial
    for dim in est.input_bbox_zyx:
        assert dim > 0
    for dim in est.intermediate_zyx:
        assert dim > 0

    # The host-side input is sized to the raw dtype (uint16 -> 2 bytes)
    assert est.host_input_bytes == int(np.prod(est.input_bbox_zyx)) * 2
    # All GPU-side buffers are sized to float32 regardless of raw dtype
    assert est.gpu_input_bytes == int(np.prod(est.input_bbox_zyx)) * 4
    assert est.gpu_intermediate_bytes == int(np.prod(est.intermediate_zyx)) * 4
    assert est.gpu_output_bytes == int(np.prod(est.output_crop_zyx)) * 4

    # GPU working set sums the coexisting buffers (input + intermediate) with
    # the safety factor; output is a view into intermediate and not double-counted.
    assert est.gpu_working_set == int(
        (est.gpu_input_bytes + est.gpu_intermediate_bytes) * est.safety_factor
    )
    assert est.host_working_set == int(est.host_input_bytes * est.safety_factor)


def test_estimate_intermediate_dominates_output_for_thick_stack():
    # The deskewed shape grows along the shear axis; for a deep stack the
    # intermediate buffer should be substantially larger than the final crop.
    raw = np.zeros((100, 60, 60), dtype=np.uint16)
    roi = [[[0, 0], [0, 30], [30, 30], [30, 0]]]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, roi, tmpdir)
        est = estimate_roi(lattice, 0)
    assert est.gpu_intermediate_bytes > est.gpu_output_bytes


def test_max_single_allocation_is_largest_buffer():
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    roi = [[[0, 0], [0, 40], [40, 40], [40, 0]]]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, roi, tmpdir)
        est = estimate_roi(lattice, 0)
    assert est.max_single_allocation == max(est.gpu_input_bytes, est.gpu_intermediate_bytes)


def test_estimate_pipeline_summary_consistent_with_per_roi():
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    rois = [
        [[0, 0], [0, 40], [40, 40], [40, 0]],
        [[5, 5], [5, 35], [25, 35], [25, 5]],
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, rois, tmpdir)
        est = estimate_pipeline(lattice, n_workers=2, safety_factor=2.0)

    assert len(est.rois) == 2
    assert est.n_workers == 2
    assert est.worker_peak_bytes == max(r.gpu_working_set for r in est.rois)
    assert est.total_gpu_bytes == est.worker_peak_bytes * 2


def test_recommended_workers_capped_by_roi_count():
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    rois = [
        [[0, 0], [0, 30], [30, 30], [30, 0]],
        [[5, 5], [5, 35], [25, 35], [25, 5]],
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, rois, tmpdir)
        # Even with plenty of VRAM, never recommend more workers than ROIs
        est = estimate_pipeline(lattice, n_workers=20)
    assert est.recommended_workers <= len(rois)


def test_recommended_workers_caps_at_one_when_per_buffer_violated(monkeypatch):
    # Force the max-alloc cap below any sensible buffer; per-buffer violation
    # is "no worker count can fix this" -> recommendation = 0
    from lls_core import estimate as est_mod
    monkeypatch.setattr(est_mod, "get_max_allocation_size", lambda: 1)
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    roi = [[[0, 0], [0, 30], [30, 30], [30, 0]]]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, roi, tmpdir)
        est = estimate_pipeline(lattice, n_workers=4)
    assert est.per_buffer_violators
    assert est.recommended_workers == 0


def test_slurm_cpu_cap_respected_in_recommendation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    # Three small ROIs; without the cap, recommendation could be 3.
    rois = [
        [[0, 0], [0, 20], [20, 20], [20, 0]],
        [[0, 0], [0, 20], [20, 20], [20, 0]],
        [[0, 0], [0, 20], [20, 20], [20, 0]],
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, rois, tmpdir)
        est = estimate_pipeline(lattice, n_workers=8)
    assert est.recommended_workers <= 2


def test_estimate_pipeline_safety_factor_scales_working_set():
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    roi = [[[0, 0], [0, 40], [40, 40], [40, 0]]]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, roi, tmpdir)
        small = estimate_pipeline(lattice, n_workers=1, safety_factor=1.0)
        large = estimate_pipeline(lattice, n_workers=1, safety_factor=3.0)

    assert large.worker_peak_bytes == 3 * small.worker_peak_bytes
    assert large.host_worker_peak_bytes == 3 * small.host_worker_peak_bytes


def test_estimate_pipeline_no_crop_returns_empty_rois():
    raw = np.zeros((10, 10, 10), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(1, 1, 1),
            save_name="t",
            save_dir=tmpdir,
        )
        est = estimate_pipeline(lattice, n_workers=2)
    assert est.rois == []
    assert est.worker_peak_bytes == 0


def test_estimate_report_formats_without_error():
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    roi = [[[0, 0], [0, 40], [40, 40], [40, 0]]]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, roi, tmpdir)
        est = estimate_pipeline(lattice, n_workers=2)
    report = est.format_report()
    assert "Lattice pre-flight" in report
    assert "Workers" in report


def test_slurm_memory_env_caps_host_estimate(monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16")  # 16 MiB - artificially tiny
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    roi = [[[0, 0], [0, 40], [40, 40], [40, 0]]]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, roi, tmpdir)
        est = estimate_pipeline(lattice, n_workers=4)
    # Honoured: host budget should be at most the SLURM cap.
    assert est.host_available_bytes is not None
    assert est.host_available_bytes <= 16 * 1024 * 1024


def test_gpu_helpers_return_sane_or_none():
    # Either return None on a headless host or a positive integer on a real device.
    g = get_global_mem_size()
    m = get_max_allocation_size()
    assert g is None or g > 0
    assert m is None or m > 0


# --- parallel save end-to-end ----------------------------------------------

def _roi(y0: int, x0: int, y1: int, x1: int) -> List[List[int]]:
    return [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]


def test_parallel_save_matches_serial_output():
    # Build two ROIs distinct enough to discriminate good output
    raw = np.zeros((30, 60, 60), dtype=np.uint16)
    raw[5:15, 5:25, 5:25] = 50
    raw[5:15, 30:50, 30:50] = 80
    rois = [_roi(0, 0, 30, 30), _roi(0, 30, 30, 60)]

    with tempfile.TemporaryDirectory() as ser_dir, tempfile.TemporaryDirectory() as par_dir:
        serial = _make_lattice(raw, rois, ser_dir, process_parallel=1)
        serial.save()
        parallel = _make_lattice(raw, rois, par_dir, process_parallel=2)
        parallel.save()

        ser_files = sorted(p.name for p in Path(ser_dir).iterdir())
        par_files = sorted(p.name for p in Path(par_dir).iterdir())
        # Same set of files produced
        assert ser_files == par_files, (ser_files, par_files)

        # Byte-compare each output
        for name in ser_files:
            ser_bytes = (Path(ser_dir) / name).read_bytes()
            par_bytes = (Path(par_dir) / name).read_bytes()
            assert ser_bytes == par_bytes, f"Parallel output diverged for {name}"


def test_use_parallel_falls_back_when_single_roi():
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    roi = [_roi(0, 0, 30, 30)]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, roi, tmpdir, process_parallel=4)
        # With only one ROI we should not bother spinning up workers
        assert lattice._use_parallel_roi_processing() is False


@pytest.mark.parametrize(
    "cli_args",
    [
        pytest.param(["--roi-subset", "0,1,2"], id="comma"),
        pytest.param(["--roi-subset", "0", "--roi-subset", "1", "--roi-subset", "2"], id="repeated"),
        pytest.param(["--roi-subset", "0,1", "--roi-subset", "2"], id="mixed"),
        pytest.param(["--roi-subset", "0, 1 , 2"], id="whitespace"),
    ],
)
def test_cli_roi_subset_accepts_commas_and_repeated(cli_args):
    """The CLI should accept --roi-subset as comma-separated, repeated, or mixed."""
    from typer.testing import CliRunner
    import lls_core.cmds.__main__ as m

    captured: dict = {}

    def fake_parse_obj(d):
        captured.update(d)
        raise SystemExit(0)

    runner = CliRunner()
    original = m.LatticeData.parse_obj
    m.LatticeData.parse_obj = staticmethod(fake_parse_obj)  # type: ignore[assignment]
    try:
        result = runner.invoke(
            m.app,
            ["dummy.tif", "--save-dir", "/tmp", "--save-name", "x", *cli_args],
            catch_exceptions=False,
        )
    finally:
        m.LatticeData.parse_obj = original  # type: ignore[assignment]
    assert result.exit_code == 0
    assert captured["crop"]["roi_subset"] == [0, 1, 2]


def test_cli_roi_subset_rejects_non_integer():
    """A non-integer in the comma list should fail fast, not crash later."""
    from typer.testing import CliRunner
    import lls_core.cmds.__main__ as m

    runner = CliRunner()
    result = runner.invoke(
        m.app,
        ["dummy.tif", "--save-dir", "/tmp", "--save-name", "x", "--roi-subset", "0,foo"],
        catch_exceptions=False,
    )
    assert result.exit_code != 0


def test_use_parallel_disabled_when_cropping_off():
    raw = np.zeros((10, 10, 10), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(1, 1, 1),
            save_name="t",
            save_dir=tmpdir,
            process_parallel=4,
        )
        assert lattice._use_parallel_roi_processing() is False
