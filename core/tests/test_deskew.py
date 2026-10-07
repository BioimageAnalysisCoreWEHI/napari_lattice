#filename and function name should start with "test_" when using pytest
import pyclesperanto_prototype as cle 
import numpy as np 
from lls_core.models.lattice_data import LatticeData
from xarray import DataArray
import tempfile

def test_deskew():

    raw = np.zeros((5,5,5))
    raw[2,0,0] = 10
    
    deskewed = cle.deskew_y(raw,angle_in_degrees=60)
    
    #np.argwhere(deskewed>0)
    assert deskewed.shape == (4,8,5)
    assert deskewed[2,2,0] == 0.5662433505058289

def test_lattice_data_deskew():
    raw = DataArray(np.zeros((5, 5, 5)), dims=["X", "Y", "Z"])
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=raw,
            physical_pixel_sizes = (1, 1, 1),
            save_name="test",
            save_dir=tmpdir
        )
        assert lattice.derived.deskew_vol_shape == (2, 9, 5)


def test_invert_scan_direction_slice_data():
    # Flip should reverse the scan (Z) axis of the volume handed to the deskew
    from lls_core.models.deskew import DeskewParams

    raw_np = np.arange(5 * 4 * 3, dtype=float).reshape(5, 4, 3)
    raw = DataArray(raw_np, dims=["Z", "Y", "X"])

    normal = DeskewParams(input_image=raw, physical_pixel_sizes=(1, 1, 1))
    inverted = DeskewParams(input_image=raw, physical_pixel_sizes=(1, 1, 1), invert_scan_direction=True)

    # Default leaves orientation untouched; inverting reverses the Z axis
    np.testing.assert_array_equal(np.asarray(normal.get_3d_slice()), raw_np)
    np.testing.assert_array_equal(np.asarray(inverted.get_3d_slice()), raw_np[::-1])


def test_invert_scan_direction_deskew_equivalence():
    # Use an off-centre voxel as an asymmetric feature so errors in
    # scan-direction inversion (wrong flip direction) are detectable.
    raw_np = np.zeros((5, 5, 5))
    raw_np[1, 1, 1] = 10

    def deskew(array: np.ndarray, invert: bool, tmpdir: str) -> np.ndarray:
        lattice = LatticeData(
            input_image=DataArray(array, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(1, 1, 1),
            invert_scan_direction=invert,
            save_name="test",
            save_dir=tmpdir,
        )
        return np.asarray(next(iter(lattice.process().slices)).data)

    with tempfile.TemporaryDirectory() as tmpdir:
        normal = deskew(raw_np, False, tmpdir)
        inverted = deskew(raw_np, True, tmpdir)
        # Inverting the scan direction must equal reversing the raw scan axis then deskewing normally
        manual = deskew(raw_np[::-1].copy(), False, tmpdir)

    assert inverted.shape == normal.shape
    np.testing.assert_allclose(inverted, manual)
    # For an asymmetric input, the flip must actually change the result
    assert not np.allclose(inverted, normal)


def test_invert_scan_direction_display_transform():
    # "Quick Deskew" display transform (deskew_affine_transform_zyx) must
    # encode the scan flip, so applying it to  unflipped layer reproduces the
    # flipped-then-deskewed geometry used by the actual processing/preview output.
    from lls_core.models.deskew import DeskewParams

    raw_np = np.zeros((6, 6, 6))
    raw_np[1, 2, 3] = 10
    raw = DataArray(raw_np, dims=["Z", "Y", "X"])
    nz = raw_np.shape[0]

    normal = DeskewParams(input_image=raw, physical_pixel_sizes=(1, 1, 1))
    inverted = DeskewParams(input_image=raw, physical_pixel_sizes=(1, 1, 1), invert_scan_direction=True)

    m = np.asarray(normal.derived.deskew_affine_transform_zyx)
    m_flipped = np.asarray(inverted.derived.deskew_affine_transform_zyx)

    # Applying the folded transform to a raw coordinate must match applying the standard
    # transform to the scan-reversed coordinate (z -> nz - 1 - z)
    for z, y, x in [(0, 0, 0), (1, 2, 3), (5, 4, 1)]:
        folded = m_flipped @ np.array([z, y, x, 1.0])
        reference = m @ np.array([nz - 1 - z, y, x, 1.0])
        np.testing.assert_allclose(folded, reference)

    # With the flag off, the display transform must be untouched
    np.testing.assert_allclose(
        m, np.asarray(DeskewParams(input_image=raw, physical_pixel_sizes=(1, 1, 1)).derived.deskew_affine_transform_zyx)
    )


def test_invert_scan_direction_crop_equivalence():
    # Cropping is performed in deskewed space and mapped back to the (flipped) raw,
    # so an inverted-scan crop must match cropping a manually-flipped raw volume.
    from lls_core.models.crop import CropParams

    raw_np = np.zeros((5, 5, 5))
    raw_np[1, 1, 1] = 10
    # A generous ROI (clipped to bounds) that covers the deskewed plane
    roi = [[[0, 0], [0, 100], [100, 100], [100, 0]]]

    def crop_deskew(array: np.ndarray, invert: bool, nz: int, tmpdir: str) -> np.ndarray:
        lattice = LatticeData(
            input_image=DataArray(array, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(1, 1, 1),
            invert_scan_direction=invert,
            crop=CropParams(roi_list=roi, z_range=(0, nz)),
            save_name="test",
            save_dir=tmpdir,
        )
        return np.asarray(next(iter(lattice.process().slices)).data)

    with tempfile.TemporaryDirectory() as tmpdir:
        nz = LatticeData(
            input_image=DataArray(raw_np, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(1, 1, 1),
            save_name="test",
            save_dir=tmpdir,
        ).derived.deskew_vol_shape[0]

        inverted = crop_deskew(raw_np, True, nz, tmpdir)
        manual = crop_deskew(raw_np[::-1].copy(), False, nz, tmpdir)

    np.testing.assert_allclose(inverted, manual)


def test_invert_scan_direction_workflow_path():
    # The workflow path processes per-slice sub-lattices built by `iter_sublattices`.
    # `slice_data` already bakes the scan flip into each sub-lattice's input, so the
    # sub-lattice must NOT flip again (a double flip would cancel out). This guards the
    # `invert_scan_direction=False` reset in `iter_sublattices` against regressions.
    raw_np = np.zeros((5, 5, 5))
    raw_np[1, 1, 1] = 10

    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=DataArray(raw_np, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(1, 1, 1),
            invert_scan_direction=True,
            save_name="test",
            save_dir=tmpdir,
        )
        # Direct processing flips exactly once
        direct = np.asarray(next(iter(lattice.process().slices)).data)
        # Each sub-lattice already holds flipped data; processing it must flip once total
        sublattice = next(iter(lattice.iter_sublattices()))
        assert sublattice.data.invert_scan_direction is False
        via_workflow = np.asarray(next(iter(sublattice.data.process().slices)).data)

    np.testing.assert_allclose(via_workflow, direct)
def test_invert_scan_direction_crop_workflow_path():
    # Crop + flip + workflow exercised together. The workflow path copies BOTH the crop
    # and the (reset) invert flag into each sub-lattice via `iter_sublattices`, so the scan
    # flip must still be applied exactly once end-to-end. Inverting the scan must equal
    # reversing the raw scan axis manually and processing the same crop workflow normally.
    from pathlib import Path
    from lls_core.models.crop import CropParams

    workflow_path = Path(__file__).parent / "workflows" / "binarisation" / "workflow.yml"

    # Asymmetric slab, bright enough to survive the binarisation workflow threshold
    raw_np = np.zeros((20, 30, 30))
    raw_np[3:12, 6:22, 6:22] = 500

    def crop_workflow(array: np.ndarray, invert: bool, shape: tuple, tmpdir: str) -> np.ndarray:
        nz, yd, xd = shape
        lattice = LatticeData(
            input_image=DataArray(array, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(1, 1, 1),
            invert_scan_direction=invert,
            crop=CropParams(roi_list=[[[0, 0], [0, xd], [yd, xd], [yd, 0]]], z_range=(0, nz)),
            workflow=str(workflow_path),
            save_name="test",
            save_dir=tmpdir,
        )
        return np.asarray(list(lattice.process_workflow().roi_previews())[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        shape = LatticeData(
            input_image=DataArray(raw_np, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(1, 1, 1),
            save_name="test",
            save_dir=tmpdir,
        ).derived.deskew_vol_shape

        inverted = crop_workflow(raw_np, True, shape, tmpdir)
        manual = crop_workflow(raw_np[::-1].copy(), False, shape, tmpdir)
        not_inverted = crop_workflow(raw_np, False, shape, tmpdir)

    # The flipped crop workflow must match cropping a manually-reversed raw volume
    np.testing.assert_allclose(inverted, manual)
    # The workflow must actually produce content (otherwise the comparison is vacuous)
    assert (inverted > 0).any()
    # A double flip in the sub-lattice path would make this equal to the un-inverted result
    assert not np.allclose(inverted, not_inverted)

