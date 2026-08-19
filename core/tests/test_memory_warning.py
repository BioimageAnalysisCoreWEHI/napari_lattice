"""
Tests for the deskew memory warning: predicting that a deskewed volume will not fit
on the GPU, and translating the opaque OpenCL failure when it does not.

The warning is advisory by design - these tests pin that it never blocks processing.
"""
from __future__ import annotations

import logging
import tempfile

import numpy as np
import pytest
from xarray import DataArray

from lls_core.estimate import (
    DeskewMemoryError,
    DeskewVolumeEstimate,
    DeviceLimits,
    estimate_pipeline,
    is_memory_error,
    memory_errors_explained,
    reset_warning_history,
    warn_if_deskew_may_not_fit,
)
from lls_core.models.crop import CropParams
from lls_core.models.lattice_data import LatticeData

GIB = 1024 ** 3


@pytest.fixture(autouse=True)
def _forget_warnings():
    """The warning dedupe is process-global, so clear it around every test."""
    reset_warning_history()
    yield
    reset_warning_history()


@pytest.fixture
def tiny_gpu(monkeypatch):
    """A device too small for anything, so the warning path is exercised."""
    from lls_core import estimate as est_mod
    monkeypatch.setattr(est_mod, "get_max_allocation_size", lambda: 100 * 1024)
    monkeypatch.setattr(est_mod, "get_global_mem_size", lambda: 8 * GIB)
    monkeypatch.setattr(est_mod, "get_host_available_bytes", lambda: 8 * GIB)


def _estimate(output_zyx, input_zyx=(100, 200, 300), max_alloc=4 * GIB,
              global_mem=8 * GIB, host=16 * GIB) -> DeskewVolumeEstimate:
    return DeskewVolumeEstimate(
        input_zyx=input_zyx,
        output_zyx=output_zyx,
        input_itemsize=2,
        output_itemsize=2,
        safety_factor=1.5,
        device=DeviceLimits(
            gpu_global_bytes=global_mem,
            gpu_max_alloc_bytes=max_alloc,
            gpu_reserve_bytes=512 * 1024 * 1024,
            host_available_bytes=host,
        ),
    )


def _make_lattice(raw: np.ndarray, tmpdir: str, **kwargs) -> LatticeData:
    return LatticeData(
        input_image=DataArray(raw, dims=["Z", "Y", "X"]),
        physical_pixel_sizes=(1, 1, 1),
        angle=30,
        skew="Y",
        save_name="test",
        save_dir=tmpdir,
        save_type="tiff",
        **kwargs,
    )


# --- the estimate -----------------------------------------------------------

def test_large_volume_does_not_overflow_to_negative_bytes():
    """numpy's default integer is 32-bit on Windows, so np.prod would wrap past
    2**31 voxels - exactly the size range this estimate exists to describe."""
    est = _estimate(input_zyx=(294, 2304, 1024), output_zyx=(294, 2304, 5184))
    assert est.voxels > 2 ** 31
    assert est.gpu_output_bytes > 0
    assert est.gpu_working_set > 0
    assert est.host_peak_bytes > 0


def test_no_warning_when_it_fits():
    est = _estimate(output_zyx=(10, 10, 10), input_zyx=(10, 10, 10))
    assert est.fits_gpu is True
    assert est.warnings() == []


def test_per_allocation_cap_fails_even_with_free_vram():
    """The cap is per buffer: plenty of total VRAM does not make an oversized
    single allocation legal."""
    est = _estimate(output_zyx=(294, 2304, 5184), max_alloc=2 * GIB, global_mem=64 * GIB)
    assert est.exceeds_max_alloc is True
    warning = est.warnings()[0]
    assert "294 x 2304 x 5184" in warning       # the size the user needs to know
    assert "can only handle" in warning


def test_unknown_device_limits_produce_no_warning():
    """A device we could not query is not evidence of a problem."""
    est = _estimate(output_zyx=(294, 2304, 5184), max_alloc=None, global_mem=None, host=None)
    assert est.fits_gpu is None
    assert est.warnings() == []


def test_host_shortfall_is_reported_with_the_size():
    est = _estimate(output_zyx=(200, 1024, 1024), max_alloc=64 * GIB,
                    global_mem=64 * GIB, host=1 * GIB)
    assert est.fits_gpu is True
    warning = est.warnings()[0]
    assert "200 x 1024 x 1024" in warning
    assert "operating system" in warning


def test_no_crop_report_describes_the_volume_instead_of_zero_rois():
    """Without cropping there are no ROIs to size, but the whole-volume deskew
    still has to fit; an all-zeroes worker summary would read as 'nothing to
    worry about' for exactly the images that fail."""
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, tmpdir)
        est = estimate_pipeline(lattice, n_workers=1)
    assert est.rois == []
    assert est.deskew_volume is not None
    assert est.deskew_volume.output_zyx == tuple(lattice.derived.deskew_vol_shape)
    assert "Deskew estimate (no cropping)" in est.format_report()


# --- warning emission -------------------------------------------------------

def test_warns_at_model_construction(tiny_gpu, caplog):
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    with caplog.at_level(logging.WARNING, logger="lls_core.estimate"):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_lattice(raw, tmpdir)
    assert any("can only handle" in r.message for r in caplog.records)


def test_repeated_construction_warns_once_even_as_free_ram_drifts(monkeypatch, caplog):
    """Model construction repeats per sublattice and per ROI worker with the same
    geometry, so the same size must not be reported once per timepoint. The dedupe is
    keyed on the deskewed shape, not the message text, because the host-RAM message
    quotes currently-free RAM - which drifts precisely while a big run is allocating."""
    from lls_core import estimate as est_mod
    monkeypatch.setattr(est_mod, "get_max_allocation_size", lambda: 100 * 1024)
    monkeypatch.setattr(est_mod, "get_global_mem_size", lambda: 8 * GIB)
    # Below the ~1.0 MiB this deskew needs on the host, and drifting, so the host
    # message differs every time while the geometry does not.
    free = iter([900_000, 890_000, 800_000, 500_000])
    monkeypatch.setattr(est_mod, "get_host_available_bytes", lambda: next(free))

    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    with caplog.at_level(logging.WARNING, logger="lls_core.estimate"):
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(4):
                _make_lattice(raw, tmpdir)
    # One GPU warning and one host warning, emitted once between them - not once per build.
    assert len(caplog.records) == 2


def test_safety_factor_is_honoured(monkeypatch, caplog):
    """`memory_safety_factor` is documented as the knob to turn up after OOM crashes,
    so the warning must agree with `lls estimate` about whether a run fits."""
    from lls_core import estimate as est_mod
    # A GPU budget of 1 MB: above the ~665 KB this deskew needs at a safety factor of
    # 1.0, below the ~5.3 MB it needs at 8.0.
    monkeypatch.setattr(est_mod, "get_max_allocation_size", lambda: 64 * GIB)
    monkeypatch.setattr(est_mod, "get_global_mem_size", lambda: 1_000_000 + 512 * 1024 * 1024)
    monkeypatch.setattr(est_mod, "get_host_available_bytes", lambda: 8 * GIB)

    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmpdir:
        with caplog.at_level(logging.WARNING, logger="lls_core.estimate"):
            _make_lattice(raw, tmpdir, memory_safety_factor=1.0)
        lenient = len(caplog.records)
        caplog.clear()
        reset_warning_history()
        with caplog.at_level(logging.WARNING, logger="lls_core.estimate"):
            _make_lattice(raw, tmpdir, memory_safety_factor=8.0)
        strict = len(caplog.records)
    assert lenient == 0 and strict == 1


@pytest.mark.parametrize("kwargs, why", [
    ({"crop": CropParams(roi_list=[[[0, 0], [0, 40], [40, 40], [40, 0]]], z_range=(0, 20))},
     "cropping deskews each ROI's bounding box, never the whole volume"),
    ({"save_mip": True},
     "MIP output projects straight from the raw data"),
])
def test_no_warning_for_paths_that_never_allocate_the_volume(tiny_gpu, caplog, kwargs, why):
    """Warning about a buffer that is never allocated would send the user chasing a
    problem they do not have - and, for cropping, tell them to do what they already are."""
    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    with caplog.at_level(logging.WARNING, logger="lls_core.estimate"):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_lattice(raw, tmpdir, **kwargs)
    assert [r for r in caplog.records if "can only handle" in r.message] == [], why


def test_warning_helper_swallows_a_broken_estimate(monkeypatch):
    """A failed estimate must not stop a job that might well have succeeded."""
    from lls_core import estimate as est_mod

    def explode():
        raise RuntimeError("no device")

    monkeypatch.setattr(est_mod, "get_max_allocation_size", explode)
    assert warn_if_deskew_may_not_fit((1, 2, 3), (1, 2, 6), np.dtype("uint16")) == []


def test_oversized_image_is_warned_about_but_still_processed(tiny_gpu, caplog):
    """The estimate is a prediction, not a gate: a user who knows their hardware
    must still be able to try."""
    raw = np.random.randint(0, 100, (30, 50, 50), dtype=np.uint16)
    with caplog.at_level(logging.WARNING, logger="lls_core.estimate"):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = list(_make_lattice(raw, tmpdir).process().slices)
    assert any("can only handle" in r.message for r in caplog.records)
    assert len(results) == 1


# --- translating the OpenCL failure -----------------------------------------

@pytest.mark.parametrize("exc, recognised", [
    (RuntimeError("clEnqueueReadBuffer failed: OUT_OF_RESOURCES"), True),
    (RuntimeError("clEnqueueWriteBuffer failed: OUT_OF_RESOURCES"), True),
    (RuntimeError("clCreateBuffer failed: MEM_OBJECT_ALLOCATION_FAILURE"), True),
    (RuntimeError("clCreateBuffer failed: INVALID_BUFFER_SIZE"), True),
    (MemoryError(), True),
    (RuntimeError("clBuildProgram failed: BUILD_PROGRAM_FAILURE"), False),
    (ValueError("angle must be positive"), False),
])
def test_only_memory_failures_are_recognised(exc, recognised):
    assert is_memory_error(exc) is recognised


def test_context_manager_passes_other_errors_through_unchanged():
    with pytest.raises(ValueError, match="unrelated"):
        with memory_errors_explained(None, "Deskewing"):
            raise ValueError("unrelated")


def test_processing_translates_the_opencl_error(monkeypatch):
    """End to end: the error a user actually sees names the image size instead of
    only an OpenCL status code."""
    import pyclesperanto_prototype as cle

    def out_of_resources(*args, **kwargs):
        raise RuntimeError("clEnqueueReadBuffer failed: OUT_OF_RESOURCES")

    monkeypatch.setattr(cle, "pull_zyx", out_of_resources)

    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, tmpdir)
        with pytest.raises(DeskewMemoryError) as excinfo:
            list(lattice.process().slices)

    message = str(excinfo.value)
    assert "ran out of GPU or host memory" in message
    # The deskewed size, which the raw OpenCL error never mentions
    z, y, x = lattice.derived.deskew_vol_shape
    assert f"{z} x {y} x {x}" in message
    # The original error survives, both in the text and as the chained cause
    assert "clEnqueueReadBuffer failed: OUT_OF_RESOURCES" in message
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_crop_failure_reports_the_roi_size_not_the_whole_volume(monkeypatch):
    """A failed ROI must be described by its own deskewed subvolume. Quoting the
    whole-volume size would name a buffer that was never allocated, and telling a
    user who is already cropping to 'crop to a region of interest' is no help."""
    import lls_core.models.lattice_data as ld
    from lls_core.estimate import estimate_roi

    def out_of_resources(*args, **kwargs):
        raise RuntimeError("clEnqueueWriteBuffer failed: OUT_OF_RESOURCES")

    monkeypatch.setattr(ld, "crop_volume_deskew", out_of_resources)

    raw = np.zeros((30, 50, 50), dtype=np.uint16)
    rois = [[[0, 0], [0, 30], [30, 30], [30, 0]]]
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = _make_lattice(raw, tmpdir, crop=CropParams(roi_list=rois, z_range=(0, 20)))
        roi_shape = estimate_roi(lattice, 0).intermediate_zyx
        with pytest.raises(DeskewMemoryError) as excinfo:
            list(lattice.process().slices)

    message = str(excinfo.value)
    assert "ROI 0" in message
    assert "{} x {} x {}".format(*roi_shape) in message
    whole_volume = "{} x {} x {}".format(*lattice.derived.deskew_vol_shape)
    assert whole_volume not in message, "must not quote a buffer that is never allocated"
