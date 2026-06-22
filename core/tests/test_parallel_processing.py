"""
Tests for the parallel-ROI processing and memory estimator.
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
    _parse_slurm_mem_bytes,
    chunk_roi_subset,
    estimate_pipeline,
    estimate_roi,
    get_roi_bboxes,
)
from lls_core.models.crop import CropParams
from lls_core.models.lattice_data import LatticeData

# GPU detection for the deconvolution end-to-end test (mirrors test_deconvolution.py).
try:
    import pyclesperanto_prototype as _cle
    _gpu_devices = _cle.available_device_names(dev_type="gpu")
except Exception:
    _gpu_devices = []
try:
    import pycudadecon._libwrap  # noqa: F401
    _cuda_decon_available = True
except (FileNotFoundError, ModuleNotFoundError):
    _cuda_decon_available = False


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
    # Fewer ROIs than workers must not create empty chunks
    assert chunk_roi_subset([0, 1], 4) == [[0], [1]]


# --- estimator pure shape math ---------------------------------------------

def test_estimate_roi_returns_nonzero_bboxes():
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    roi = [[[0, 0], [0, 40], [40, 40], [40, 0]]]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, roi, tmpdir)
        est = estimate_roi(lattice, 0)

    assert est.roi_index == 0
    # Both bboxes are non-trivial
    for dim in est.input_bbox_zyx:
        assert dim > 0
    for dim in est.intermediate_zyx:
        assert dim > 0

    # The host-side input is sized to the raw dtype (uint16 -> 2 bytes)
    assert est.host_input_bytes == int(np.prod(est.input_bbox_zyx)) * 2
    # All GPU-side buffers are sized to float32 regardless of raw dtype
    assert est.gpu_input_bytes == int(np.prod(est.input_bbox_zyx)) * 4
    assert est.gpu_intermediate_bytes == int(np.prod(est.intermediate_zyx)) * 4

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
        _input_bbox, intermediate_zyx, output_crop_zyx = get_roi_bboxes(lattice, 0)
    assert int(np.prod(intermediate_zyx)) > int(np.prod(output_crop_zyx))


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
    assert "Memory estimate" in report
    assert "recommended" in report


@pytest.mark.parametrize(
    "value,expected",
    [
        ("16384", 16384 * 1024 ** 2),   # bare number is MiB
        ("16384M", 16384 * 1024 ** 2),
        ("16G", 16 * 1024 ** 3),
        ("16g", 16 * 1024 ** 3),        # case-insensitive
        ("2T", 2 * 1024 ** 4),
        (" 16G ", 16 * 1024 ** 3),      # whitespace tolerated
        ("0", None),                    # SLURM 0 == all node memory, i.e. no cap
        ("garbage", None),
        ("", None),
    ],
)
def test_parse_slurm_mem_bytes(value, expected):
    assert _parse_slurm_mem_bytes(value) == expected


@pytest.mark.parametrize("env_val", ["16", "16M"])  # plain MiB and suffixed must both cap
def test_slurm_memory_env_caps_host_estimate(monkeypatch, env_val):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", env_val)  # 16 MiB - artificially tiny
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    roi = [[[0, 0], [0, 40], [40, 40], [40, 0]]]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, roi, tmpdir)
        est = estimate_pipeline(lattice, n_workers=4)
    # Honoured: host budget should be at most the SLURM cap.
    assert est.host_available_bytes is not None
    assert est.host_available_bytes <= 16 * 1024 * 1024


# --- parallel save end-to-end ----------------------------------------------

def _roi(y0: int, x0: int, y1: int, x1: int) -> List[List[int]]:
    return [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]


# --- input source provenance ------------------------------------------------

def test_input_image_path_captured_from_path(rbc_tiny):
    # A path input records its source (so workers can re-open it lazily); an
    # in-memory array has no path and is sent to workers directly.
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmpdir:
        from_path = LatticeData(
            input_image=str(rbc_tiny), save_name="t", save_dir=tmpdir, save_type="tiff",
            crop=CropParams(roi_list=[_roi(0, 0, 60, 60)], z_range=(0, 5)),
        )
        from_array = _make_lattice(raw, [_roi(0, 0, 30, 30)], tmpdir)
    assert from_path.input_image_path is not None and from_path.input_image_path.exists()
    assert from_array.input_image_path is None


def test_input_image_path_survives_alongside_array(rbc_tiny):
    # GUI single-file case: an explicit path passed with an array must be retained,
    # so the dispatcher uses lazy reload instead of materializing the whole volume.
    from bioio import BioImage
    arr = BioImage(str(rbc_tiny)).xarray_dask_data
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=arr, input_image_path=Path(str(rbc_tiny)),
            physical_pixel_sizes=(1, 1, 1),
            save_name="t", save_dir=tmpdir, save_type="tiff",
            crop=CropParams(roi_list=[_roi(0, 0, 60, 60)], z_range=(0, 5)),
        )
    assert lattice.input_image_path == Path(str(rbc_tiny))


# --- parallel save end-to-end (parallel output must equal serial) -----------

def _numpy_kwargs(rbc_tiny):
    raw = np.zeros((30, 60, 60), dtype=np.uint16)
    raw[5:15, 5:25, 5:25] = 50
    raw[5:15, 30:50, 30:50] = 80
    return dict(
        input_image=DataArray(raw, dims=["Z", "Y", "X"]),
        physical_pixel_sizes=(1, 1, 1),
        crop=CropParams(roi_list=[_roi(0, 0, 30, 30), _roi(0, 30, 30, 60)], z_range=(0, 20)),
    )


def _file_path_kwargs(rbc_tiny):
    return dict(
        input_image=str(rbc_tiny),
        crop=CropParams(roi_list=[_roi(0, 0, 60, 60), _roi(20, 20, 100, 100)], z_range=(0, 5)),
    )


def _dask_no_path_kwargs(rbc_tiny):
    from bioio import BioImage
    return dict(
        input_image=BioImage(str(rbc_tiny)).xarray_dask_data,
        physical_pixel_sizes=(1, 1, 1),
        crop=CropParams(roi_list=[_roi(0, 0, 60, 60), _roi(20, 20, 100, 100)], z_range=(0, 5)),
    )


@pytest.mark.parametrize("make_kwargs", [
    pytest.param(_numpy_kwargs, id="numpy_materialize"),
    pytest.param(_file_path_kwargs, id="file_path_lazy_reload"),
    pytest.param(_dask_no_path_kwargs, id="dask_no_path_materialize"),
])
def test_parallel_save_matches_serial(make_kwargs, rbc_tiny):
    """
    Parallel ROI save must be byte-identical to serial across input variants:
    in-memory numpy (materialize passthrough), a file path (lazy per-crop reload),
    and an in-memory dask array with no path (materialize). The numpy-only case
    would pickle fine and hide the unpicklable lazy-reader bugs, hence the file
    and dask variants.
    """
    def build(save_dir: str, parallel: int) -> LatticeData:
        return LatticeData(
            **make_kwargs(rbc_tiny),
            save_name="cmp", save_dir=save_dir, save_type="tiff",
            process_parallel=parallel,
        )

    with tempfile.TemporaryDirectory() as ser_dir, tempfile.TemporaryDirectory() as par_dir:
        build(ser_dir, 1).save()
        build(par_dir, 2).save()
        ser_files = sorted(p.name for p in Path(ser_dir).iterdir())
        par_files = sorted(p.name for p in Path(par_dir).iterdir())
        assert ser_files == par_files, (ser_files, par_files)
        for name in ser_files:
            assert (Path(ser_dir) / name).read_bytes() == (Path(par_dir) / name).read_bytes(), name


def test_parallel_isolates_hard_worker_failure(monkeypatch):
    """
    A hard worker death in one chunk (surfaced as BrokenProcessPool) must fail only
    that chunk: sibling chunks run in their own pools and still write their output,
    and the run raises so the failure is not mistaken for success.
    """
    from concurrent.futures import BrokenExecutor
    import lls_core.models.lattice_data as ld

    raw = np.zeros((30, 90, 90), dtype=np.uint16)
    rois = [_roi(0, 0, 30, 30), _roi(0, 30, 30, 60), _roi(0, 60, 30, 90)]

    original = ld._run_chunk_isolated

    def flaky(lattice, roi_indices):
        # ROI index 1 simulates a hard worker death; the others run for real.
        if 1 in roi_indices:
            raise BrokenExecutor("simulated hard worker death (e.g. OOM kill)")
        return original(lattice, roi_indices)

    monkeypatch.setattr(ld, "_run_chunk_isolated", flaky)

    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, rois, tmpdir, process_parallel=3)
        with pytest.raises(RuntimeError):
            lattice.save()

        produced = sorted(p.name for p in Path(tmpdir).iterdir())
        # The two healthy ROIs still produced output despite the sibling crash
        assert any("ROI_0" in n for n in produced), produced
        assert any("ROI_2" in n for n in produced), produced
        # The poisoned ROI did not
        assert not any("ROI_1" in n for n in produced), produced


def test_use_parallel_falls_back_when_single_roi():
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    roi = [_roi(0, 0, 30, 30)]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, roi, tmpdir, process_parallel=4)
        # With only one ROI we should not bother spinning up workers
        assert lattice._use_parallel_roi_processing() is False


def test_dispatch_payload_materializes_lazy_psf():
    """
    A deconvolution PSF loaded from a path is a lazy (dask-backed) bioio reader; the
    dispatch payload must materialize it so workers can receive it by pickle. Runs
    without a GPU - only the pickle-time preparation is exercised, not the decon.
    """
    import pickle
    import dask.array as da
    from importlib_resources import as_file
    from lls_core.sample import resources
    from lls_core.models.deconvolution import DeconvolutionParams

    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    rois = [_roi(0, 0, 30, 30), _roi(0, 20, 30, 50)]
    with as_file(resources / "psfs/zeiss_simulated/488.tif") as psf_path:
        with tempfile.TemporaryDirectory() as tmpdir:
            lattice = LatticeData(
                input_image=DataArray(raw, dims=["Z", "Y", "X"]),
                physical_pixel_sizes=(1, 1, 1),
                save_name="d", save_dir=tmpdir, save_type="tiff",
                crop=CropParams(roi_list=rois, z_range=(0, 20)),
                deconvolution=DeconvolutionParams(psf=[str(psf_path)]),
                process_parallel=2,
            )
            # Raw PSF is lazy (dask); the dispatch payload materializes it...
            assert isinstance(lattice.deconvolution.psf[0].data, da.Array)
            payload = lattice._dispatch_payload()
            assert not isinstance(payload.deconvolution.psf[0].data, da.Array)
            # ...so the payload (image + PSF) pickles cleanly for the workers.
            pickle.dumps(payload)


@pytest.mark.skipif(len(_gpu_devices) < 1, reason="No GPU detected")
@pytest.mark.skipif(not _cuda_decon_available, reason="pycudadecon not installed")
def test_parallel_decon_matches_serial(rbc_tiny):
    """
    End-to-end: deconvolution + parallel ROI save must run (PSF materialized for the
    workers) and be byte-identical to serial. GPU-gated, so CI skips it.
    """
    from importlib_resources import as_file
    from lls_core.sample import resources
    from lls_core.models.deconvolution import DeconvolutionParams

    rois = [_roi(0, 0, 120, 120), _roi(40, 40, 200, 200)]

    def build(psf_path, save_dir, parallel):
        return LatticeData(
            input_image=str(rbc_tiny),
            save_name="decon", save_dir=save_dir, save_type="tiff",
            crop=CropParams(roi_list=rois, z_range=(0, 55)),
            deconvolution=DeconvolutionParams(
                decon_processing="cuda_gpu", psf=[str(psf_path)], decon_num_iter=2),
            process_parallel=parallel,
        )

    with as_file(resources / "psfs/zeiss_simulated/488.tif") as psf, \
         tempfile.TemporaryDirectory() as ser_dir, tempfile.TemporaryDirectory() as par_dir:
        build(psf, ser_dir, 1).save()
        build(psf, par_dir, 2).save()
        ser_files = sorted(p.name for p in Path(ser_dir).iterdir())
        par_files = sorted(p.name for p in Path(par_dir).iterdir())
        assert ser_files == par_files, (ser_files, par_files)
        for name in ser_files:
            assert (Path(ser_dir) / name).read_bytes() == (Path(par_dir) / name).read_bytes(), name


def test_resolve_worker_count_explicit():
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    rois = [_roi(0, 0, 30, 30), _roi(0, 20, 30, 50)]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, rois, tmpdir, process_parallel=2)
        assert lattice._resolve_worker_count() == 2


def test_resolve_worker_count_auto_uses_estimate():
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    rois = [_roi(0, 0, 30, 30), _roi(0, 20, 30, 50)]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, rois, tmpdir, process_parallel=0)
        est = estimate_pipeline(lattice, n_workers=1)
        n = lattice._resolve_worker_count()
        assert n == max(1, est.recommended_workers)
        assert n >= 1


def test_auto_resolves_serial_when_workflow_attached():
    # The estimate can't size workflow memory, so auto must fall back to serial
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    rois = [_roi(0, 0, 30, 30), _roi(0, 20, 30, 50)]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, rois, tmpdir, process_parallel=0)
        # Inject a workflow without full construction (bypasses validation)
        object.__setattr__(lattice, "workflow", object())
        assert lattice._resolve_worker_count() == 1


def test_cli_defaults_process_parallel_to_auto():
    """A CLI run with no --process-parallel should default to auto (0)."""
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
        runner.invoke(
            m.app,
            ["dummy.tif", "--save-dir", "/tmp", "--save-name", "x"],
            catch_exceptions=False,
        )
    finally:
        m.LatticeData.parse_obj = original  # type: ignore[assignment]
    assert captured.get("process_parallel") == 0


def test_cli_explicit_process_parallel_respected():
    """An explicit --process-parallel must override the auto default."""
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
        runner.invoke(
            m.app,
            ["dummy.tif", "--save-dir", "/tmp", "--save-name", "x", "--process-parallel", "3"],
            catch_exceptions=False,
        )
    finally:
        m.LatticeData.parse_obj = original  # type: ignore[assignment]
    assert captured.get("process_parallel") == 3


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
