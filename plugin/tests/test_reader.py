"""
Tests for napari_lattice.reader's channel-axis splitting decision and the reader -> GUI
dimension-order contract.

bioio fabricates a full ``TCZYX`` order for plain TIFFs (padding missing axes with
size-1 and auto-naming channels ``Channel:i:j``). The reader must only split a layer
along its channel axis when that axis is genuine; otherwise single-channel stacks
collapse in dimensionality and frame axes that bioio mislabelled "C" get carved into
spurious channels - which is what hides the TCZYX/CTZYX dimension-order options the user
needs to correct the interpretation.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile
from bioio import BioImage
from importlib_resources import as_file

from lls_core.sample import resources
from napari_lattice.fields import dimension_order_options
from napari_lattice.reader import _AUTO_CHANNEL_NAME, bioio_reader


def _plain_tiff(tmp_path, shape) -> str:
    p = tmp_path / "plain.tif"
    tifffile.imwrite(str(p), np.zeros(shape, dtype=np.uint16))
    return str(p)


@pytest.mark.parametrize("on_disk", [
    pytest.param((5, 1, 30, 16, 16), id="single_channel"),       # bioio -> TCZYX, C=1
    pytest.param((4, 30, 16, 16), id="frames_mislabelled_C"),    # bioio -> TCZYX, C=4
    pytest.param((3, 2, 30, 16, 16), id="multichannel_no_meta"), # placeholder names
])
def test_plain_tiff_is_not_split(tmp_path, on_disk):
    # No genuine channel metadata -> keep the layer whole so the GUI can offer the
    # full TCZYX/CTZYX dimension-order options for the user to correct.
    layers = bioio_reader(_plain_tiff(tmp_path, on_disk))
    assert len(layers) == 1
    _data, add_kwargs, _ = layers[0]
    assert "channel_axis" not in add_kwargs


def test_genuine_multichannel_is_split():
    # A real acquisition (a format-specific reader with descriptive channel names) is
    # still split into one layer per channel for convenient per-channel viewing.
    with as_file(resources / "LLS7_t1_ch3.czi") as p:
        image = BioImage(str(p))
        layers = bioio_reader(str(p))
    _data, add_kwargs, _ = layers[0]
    assert add_kwargs["channel_axis"] == image.dims.order.index("C")
    assert add_kwargs["name"] == list(image.channel_names)
    # The channel axis is dropped from the per-channel dimension metadata.
    assert "C" not in add_kwargs["metadata"]["dimensions"]


def test_single_channel_tiff_yields_5d_dimension_options(tmp_path):
    # The reader -> GUI contract: a single-channel stack stays 5D, and the dimension
    # dropdown must then offer the 5D orders so the user can set/confirm the order.
    data, _add_kwargs, _ = bioio_reader(_plain_tiff(tmp_path, (5, 1, 30, 16, 16)))[0]
    options = dimension_order_options(len(data.shape))
    assert "TCZYX" in options and "CTZYX" in options


def test_bioio_placeholder_channel_name_format_unchanged(tmp_path):
    # Canary: our split decision relies on recognising bioio's placeholder channel
    # names for plain TIFFs. If a bioio upgrade changes this format, fail loudly here
    # rather than silently reverting to splitting every plain TIFF.
    names = BioImage(_plain_tiff(tmp_path, (3, 2, 30, 16, 16))).channel_names
    assert names and all(_AUTO_CHANNEL_NAME.match(str(n)) for n in names), names
