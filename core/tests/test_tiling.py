"""Tests for XY-tiling deskew."""
from __future__ import annotations

from unittest.mock import patch
import tempfile
from pathlib import Path

import numpy as np
import pytest

from lls_core import DeskewDirection
from lls_core.llsz_core import (
    _should_tile_on_gpu,
    deskew_xy_tiles,
    get_xy_tile_sizes,
)
from lls_core.models import LatticeData
from lls_core.models.output import SaveFileType
from lls_core.utils import get_deskewed_shape


# ---------------------------------------------------------------------------
# _should_tile_on_gpu
# ---------------------------------------------------------------------------

def test_should_tile_small_fits():
    with patch("lls_core.utils.get_max_allocation_size", return_value=2 * 1024**3), \
         patch("lls_core.utils.get_global_mem_size", return_value=8 * 1024**3):
        assert _should_tile_on_gpu((30, 64, 64), (30, 128, 64), np.float32) is False


def test_should_tile_large_exceeds():
    with patch("lls_core.utils.get_max_allocation_size", return_value=100 * 1024**2), \
         patch("lls_core.utils.get_global_mem_size", return_value=8 * 1024**3):
        assert _should_tile_on_gpu((512, 512, 512), (512, 1024, 512), np.float32) is True


# ---------------------------------------------------------------------------
# get_xy_tile_sizes
# ---------------------------------------------------------------------------

def test_tile_sizes_small_no_split():
    with patch("lls_core.utils.get_max_allocation_size", return_value=2 * 1024**3), \
         patch("lls_core.utils.get_global_mem_size", return_value=8 * 1024**3):
        tile_y, tile_x = get_xy_tile_sizes((30, 64, 64), (30, 128, 64))
        assert tile_y == 128
        assert tile_x == 64


def test_tile_sizes_large_reduced():
    with patch("lls_core.utils.get_max_allocation_size", return_value=50 * 1024**2), \
         patch("lls_core.utils.get_global_mem_size", return_value=200 * 1024**2):
        tile_y, tile_x = get_xy_tile_sizes(
            (200, 2000, 2000), (200, 4000, 2000), DeskewDirection.Y
        )
        assert tile_x < 2000 or tile_y < 4000
        assert tile_y >= 32 and tile_x >= 32


# ---------------------------------------------------------------------------
# deskew_xy_tiles
# ---------------------------------------------------------------------------

def test_deskew_xy_tiles_shape_and_content():
    """Tiled deskew produces correct shape with non-zero output."""
    rng = np.random.default_rng(42)
    vol = rng.integers(0, 1000, size=(30, 64, 64), dtype=np.uint16)
    shape, _ = get_deskewed_shape(vol, 30.0, 0.15, 0.15, 0.3, DeskewDirection.Y)
    shape = tuple(shape)

    result = deskew_xy_tiles(vol, shape, 30.0, 0.15, 0.15, 0.3, DeskewDirection.Y)
    assert result.shape == shape
    assert result.max() > 0


def test_deskew_xy_tiles_skew_x():
    rng = np.random.default_rng(42)
    vol = rng.integers(0, 1000, size=(30, 64, 64), dtype=np.uint16)
    shape, _ = get_deskewed_shape(vol, 30.0, 0.15, 0.15, 0.3, DeskewDirection.X)
    shape = tuple(shape)

    result = deskew_xy_tiles(vol, shape, 30.0, 0.15, 0.15, 0.3, DeskewDirection.X)
    assert result.shape == shape
    assert result.max() > 0


def test_deskew_xy_tiles_dtype():
    rng = np.random.default_rng(42)
    vol = rng.integers(0, 1000, size=(20, 48, 48), dtype=np.uint16)
    shape, _ = get_deskewed_shape(vol, 30.0, 0.15, 0.15, 0.3)
    result = deskew_xy_tiles(
        vol, tuple(shape), 30.0, 0.15, 0.15, 0.3,
        DeskewDirection.Y, output_dtype=np.uint16,
    )
    assert result.dtype == np.uint16


# ---------------------------------------------------------------------------
# Integration: LatticeData pipeline with forced tiling
# ---------------------------------------------------------------------------

def test_tiled_pipeline_process(rbc_tiny):
    """Full pipeline with tiling forced produces correct output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData.parse_obj({
            "input_image": rbc_tiny,
            "save_dir": tmpdir,
        })
        with patch("lls_core.llsz_core._should_tile_on_gpu", return_value=True):
            slices = list(lattice.process().slices)
        assert len(slices) > 0
        for s in slices:
            assert s.data.ndim == 3
            assert s.data.shape == tuple(lattice.derived.deskew_vol_shape)


@pytest.mark.parametrize("save_type", [SaveFileType.tiff, SaveFileType.omezarr])
def test_tiled_pipeline_save(rbc_tiny, save_type):
    """Tiled processing produces valid output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData.parse_obj({
            "input_image": rbc_tiny,
            "save_dir": tmpdir,
            "save_type": save_type,
        })
        with patch("lls_core.llsz_core._should_tile_on_gpu", return_value=True):
            lattice.save()
        assert len(list(Path(tmpdir).iterdir())) > 0
