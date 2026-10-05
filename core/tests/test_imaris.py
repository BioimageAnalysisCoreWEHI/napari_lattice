"""
Tests for the Imaris (.ims) writer.

The layout assertions here are transcribed from the reference implementation,
``writer/bpWriterHDF5.cxx`` in https://github.com/imaris/ImarisWriter, so that a
future change to :mod:`lls_core.imaris` that drifts from the Imaris conventions
fails here rather than in Imaris.
"""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from lls_core.imaris import (
    ImsWriter,
    chunk_shape,
    optimal_image_pyramid,
    resolve_imaris_dtype,
)


def read_attr(obj, name: str) -> str:
    """
    Decode an Imaris string attribute.

    Deliberately the same expression the third-party reader uses
    (``imaris_ims_file_reader``: ``str(hf[loc].attrs[attrib], encoding='ascii')``),
    so that these tests fail if we ever write real HDF5 strings instead of the
    array-of-single-characters that Imaris expects.
    """
    return str(obj.attrs[name], encoding="ascii")


class TestDtypeMapping:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.float32])
    def test_supported_dtypes_are_preserved(self, dtype):
        assert resolve_imaris_dtype(np.dtype(dtype)) == np.dtype(dtype)

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            (bool, np.uint8),
            (np.int8, np.uint8),
            (np.int16, np.uint16),
            # Label images are commonly int32; the width must survive so IDs
            # above 65535 are not collapsed.
            (np.int32, np.uint32),
            (np.float64, np.float32),
            (np.float16, np.float32),
            (np.uint64, np.uint32),
        ],
    )
    def test_unsupported_dtypes_are_promoted(self, source, expected):
        assert resolve_imaris_dtype(np.dtype(source)) == np.dtype(expected)


class TestPyramid:
    def test_small_image_is_single_level(self):
        # 32*32*8 = 8192 voxels, far under the 1 MiB / 2 byte budget.
        assert optimal_image_pyramid((8, 32, 32), 2) == [(8, 32, 32)]

    def test_levels_shrink_monotonically_and_terminate(self):
        levels = optimal_image_pyramid((109, 3840, 5112), 2)
        assert levels[0] == (109, 3840, 5112)
        assert len(levels) > 1
        for finer, coarser in zip(levels, levels[1:]):
            assert all(c <= f for c, f in zip(coarser, finer))
            assert coarser != finer
        # The last level must fit the budget, which is what makes Imaris
        # responsive when it first opens the image.
        assert np.prod(levels[-1]) <= (1024 * 1024) // 2

    def test_thin_z_is_not_reduced_below_one(self):
        # A MIP-like singleton Z must never be halved to zero.
        levels = optimal_image_pyramid((1, 2048, 2048), 2)
        assert all(level[0] == 1 for level in levels)

    def test_reduce_z_disabled_keeps_z(self):
        levels = optimal_image_pyramid((64, 1024, 1024), 2, reduce_z=False)
        assert all(level[0] == 64 for level in levels)

    def test_anisotropy_is_reduced_towards_isotropy(self):
        """
        The reference algorithm halves only axes that are still long relative to
        the others, so a very wide, thin volume loses X and Y before Z.
        """
        levels = optimal_image_pyramid((16, 2048, 2048), 2)
        # Z (16) is much shorter than X/Y (2048), so the first step must leave it
        # alone and reduce the long axes.
        assert levels[1] == (16, 1024, 1024)


class TestChunks:
    def test_chunk_never_exceeds_image(self):
        shape = (3, 7, 11)
        assert chunk_shape(shape, 2) == shape

    def test_chunk_fits_budget(self):
        chunk = chunk_shape((109, 3840, 5112), 2)
        assert np.prod(chunk) * 2 <= 1024 * 1024
        # Cube-ish, not a single degenerate plane.
        assert all(c > 1 for c in chunk)


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    return rng.integers(0, 4096, size=(16, 64, 48), dtype=np.uint16)


@pytest.fixture
def written(tmp_path, volume):
    """A 2 timepoint, 2 channel file with distinguishable content per (t, c)."""
    path = tmp_path / "test.ims"
    with ImsWriter(
        path,
        shape_tczyx=(2, 2, *volume.shape),
        dtype=volume.dtype,
        voxel_size_zyx=(2.0, 0.145, 0.145),
        channel_names=["green", "red"],
    ) as writer:
        for t in range(2):
            for c in range(2):
                writer.write_volume(t, c, (volume + (10 * t + c)).astype(np.uint16))
    return path, volume


class TestLayout:
    def test_root_attributes(self, written):
        path, _ = written
        with h5py.File(path, "r") as f:
            assert read_attr(f, "ImarisDataSet") == "ImarisDataSet"
            assert read_attr(f, "ImarisVersion") == "5.5.0"
            assert read_attr(f, "DataSetDirectoryName") == "DataSet"
            assert read_attr(f, "DataSetInfoDirectoryName") == "DataSetInfo"
            assert f.attrs["NumberOfDataSets"][0] == 1

    def test_strings_are_character_arrays(self, written):
        """
        The single most likely way to produce a file Imaris cannot read: writing
        HDF5 strings instead of ``S1`` arrays (bpWriterHDF5.cxx:670).
        """
        path, _ = written
        with h5py.File(path, "r") as f:
            raw = f["DataSetInfo/Image"].attrs["X"]
            assert raw.dtype == np.dtype("S1")
            assert raw.shape == (2,)  # "48"

    def test_group_hierarchy(self, written):
        path, _ = written
        with h5py.File(path, "r") as f:
            for t in range(2):
                for c in range(2):
                    group = f[f"DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}"]
                    assert "Data" in group
                    assert "Histogram" in group

    def test_image_section_records_level_zero_size(self, written):
        path, volume = written
        n_z, n_y, n_x = volume.shape
        with h5py.File(path, "r") as f:
            image = f["DataSetInfo/Image"]
            assert int(read_attr(image, "X")) == n_x
            assert int(read_attr(image, "Y")) == n_y
            assert int(read_attr(image, "Z")) == n_z
            assert read_attr(image, "Unit") == "um"

    def test_extents_encode_voxel_size(self, written):
        """Imaris derives voxel size from extent / image size, so this is how
        the 0.145 x 0.145 x 2.0 um spacing reaches the viewer."""
        path, volume = written
        n_z, n_y, n_x = volume.shape
        with h5py.File(path, "r") as f:
            image = f["DataSetInfo/Image"]
            ext = [float(read_attr(image, f"ExtMax{i}")) for i in range(3)]
            assert [float(read_attr(image, f"ExtMin{i}")) for i in range(3)] == [0.0, 0.0, 0.0]
            assert ext[0] / n_x == pytest.approx(0.145)
            assert ext[1] / n_y == pytest.approx(0.145)
            assert ext[2] / n_z == pytest.approx(2.0)

    def test_time_info_is_one_based(self, written):
        """
        Group names under /DataSet are 0-based but DataSetInfo/TimeInfo counts
        from 1 (bpWriterHDF5.cxx:638).
        """
        path, _ = written
        with h5py.File(path, "r") as f:
            info = f["DataSetInfo/TimeInfo"]
            assert read_attr(info, "DatasetTimePoints") == "2"
            assert read_attr(info, "FileTimePoints") == "2"
            assert "TimePoint1" in info.attrs
            assert "TimePoint2" in info.attrs
            assert "TimePoint0" not in info.attrs

    def test_channel_metadata(self, written):
        path, _ = written
        with h5py.File(path, "r") as f:
            assert read_attr(f["DataSetInfo/Channel 0"], "Name") == "green"
            assert read_attr(f["DataSetInfo/Channel 1"], "Name") == "red"
            assert read_attr(f["DataSetInfo/Channel 0"], "ColorMode") == "BaseColor"
            # Three space separated floats.
            assert len(read_attr(f["DataSetInfo/Channel 0"], "Color").split()) == 3
            low, high = read_attr(f["DataSetInfo/Channel 0"], "ColorRange").split()
            assert float(high) > float(low)


class TestData:
    def test_roundtrip_is_lossless(self, written):
        """
        Level 0 must be bit-exact: the pyramid is a preview, the base level is
        the data.
        """
        path, volume = written
        n_z, n_y, n_x = volume.shape
        with h5py.File(path, "r") as f:
            for t in range(2):
                for c in range(2):
                    data = f[f"DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}/Data"]
                    stored = data[:n_z, :n_y, :n_x]
                    np.testing.assert_array_equal(stored, volume + (10 * t + c))

    def test_dataspace_is_chunk_padded(self, written):
        """
        Imaris pads the dataspace to whole chunks and records the true size in
        ImageSize* (bpWriterHDF5.cxx:460-470). Readers rely on that, so verify
        both halves of the contract.
        """
        path, volume = written
        n_z, n_y, n_x = volume.shape
        with h5py.File(path, "r") as f:
            group = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"]
            data = group["Data"]
            chunks = data.chunks
            for axis, size in enumerate((n_z, n_y, n_x)):
                expected = -(-size // chunks[axis]) * chunks[axis]
                assert data.shape[axis] == expected
            assert int(read_attr(group, "ImageSizeX")) == n_x
            assert int(read_attr(group, "ImageSizeY")) == n_y
            assert int(read_attr(group, "ImageSizeZ")) == n_z

    def test_histogram_is_256_bin_uint64(self, written):
        path, volume = written
        with h5py.File(path, "r") as f:
            group = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"]
            hist = group["Histogram"]
            assert hist.shape == (256,)
            assert hist.dtype == np.uint64
            # Every voxel must be accounted for.
            assert hist[:].sum() == volume.size
            assert float(read_attr(group, "HistogramMin")) <= volume.min()
            assert float(read_attr(group, "HistogramMax")) >= volume.max()

    def test_pyramid_levels_are_written_for_every_tc(self, tmp_path):
        # Large enough to force more than one resolution level.
        rng = np.random.default_rng(1)
        big = rng.integers(0, 1000, size=(32, 256, 256), dtype=np.uint16)
        path = tmp_path / "pyramid.ims"
        with ImsWriter(path, shape_tczyx=(1, 1, *big.shape), dtype=big.dtype,
                       voxel_size_zyx=(1.0, 1.0, 1.0)) as writer:
            levels = writer.levels
            writer.write_volume(0, 0, big)

        assert len(levels) > 1
        with h5py.File(path, "r") as f:
            assert len(f["DataSet"]) == len(levels)
            for r, level_shape in enumerate(levels):
                group = f[f"DataSet/ResolutionLevel {r}/TimePoint 0/Channel 0"]
                assert int(read_attr(group, "ImageSizeZ")) == level_shape[0]
                assert int(read_attr(group, "ImageSizeY")) == level_shape[1]
                assert int(read_attr(group, "ImageSizeX")) == level_shape[2]

    def test_downsampling_averages_rather_than_subsamples(self, tmp_path):
        """
        A volume that is zero everywhere except alternating planes would vanish
        under naive striding; averaging must preserve the signal as a halved
        intensity.
        """
        vol = np.zeros((32, 256, 256), dtype=np.uint16)
        vol[::2] = 100
        path = tmp_path / "mean.ims"
        with ImsWriter(path, shape_tczyx=(1, 1, *vol.shape), dtype=vol.dtype,
                       voxel_size_zyx=(1.0, 1.0, 1.0)) as writer:
            levels = writer.levels
            writer.write_volume(0, 0, vol)

        # Find a level that actually reduced Z.
        reduced = next((r for r, lv in enumerate(levels) if lv[0] < levels[0][0]), None)
        if reduced is None:
            pytest.skip("Pyramid did not reduce Z for this shape")
        with h5py.File(path, "r") as f:
            group = f[f"DataSet/ResolutionLevel {reduced}/TimePoint 0/Channel 0"]
            shape = levels[reduced]
            data = group["Data"][:shape[0], :shape[1], :shape[2]]
        assert data.max() > 0, "signal was lost by subsampling instead of averaging"
        assert data.min() > 0, "every output plane should mix a bright and a dark plane"

    def test_float_input_is_preserved_as_float32(self, tmp_path):
        vol = np.linspace(0, 1, 8 * 16 * 16, dtype=np.float64).reshape(8, 16, 16)
        path = tmp_path / "float.ims"
        with ImsWriter(path, shape_tczyx=(1, 1, *vol.shape), dtype=vol.dtype,
                       voxel_size_zyx=(1.0, 1.0, 1.0)) as writer:
            writer.write_volume(0, 0, vol)
        with h5py.File(path, "r") as f:
            data = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
            assert data.dtype == np.float32
            np.testing.assert_allclose(data[:8, :16, :16], vol, rtol=1e-6)

    def test_nan_becomes_zero_for_integer_output(self, tmp_path):
        vol = np.zeros((4, 8, 8), dtype=np.float32)
        vol[0, 0, 0] = np.nan
        vol[1, 1, 1] = 5.0
        path = tmp_path / "nan.ims"
        with ImsWriter(path, shape_tczyx=(1, 1, *vol.shape), dtype=np.uint16,
                       voxel_size_zyx=(1.0, 1.0, 1.0)) as writer:
            writer.write_volume(0, 0, vol)
        with h5py.File(path, "r") as f:
            data = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
            assert data[0, 0, 0] == 0
            assert data[1, 1, 1] == 5

    def test_out_of_range_indices_rejected(self, tmp_path, volume):
        path = tmp_path / "bounds.ims"
        with ImsWriter(path, shape_tczyx=(1, 1, *volume.shape), dtype=volume.dtype,
                       voxel_size_zyx=(1.0, 1.0, 1.0)) as writer:
            with pytest.raises(IndexError):
                writer.write_volume(1, 0, volume)
            with pytest.raises(IndexError):
                writer.write_volume(0, 1, volume)
            with pytest.raises(ValueError):
                writer.write_volume(0, 0, volume[:, :, :-1])
            writer.write_volume(0, 0, volume)


class TestWiring:
    """The enum, extension and writer lookup have to agree, or a run picks the
    right format but writes it under the wrong name."""

    def test_save_type_selects_the_imaris_writer(self):
        from lls_core.models.output import SaveFileType
        from lls_core.writers import ImarisWriter

        assert SaveFileType.imaris == "imaris"

        class FakeOutput:
            save_type = SaveFileType.imaris

        from lls_core.models.lattice_data import LatticeData
        from lls_core.models.output import OutputParams

        assert OutputParams.file_extension.fget(FakeOutput()) == "ims"
        assert LatticeData.get_writer(FakeOutput()) is ImarisWriter


def test_readable_by_imaris_ims_file_reader(written):
    """
    Cross-check against an independent third-party reader if it is installed.
    Skipped when it is not, so this never blocks the suite.
    """
    ims = pytest.importorskip("imaris_ims_file_reader.ims").ims
    path, volume = written
    n_z, n_y, n_x = volume.shape
    image = ims(str(path))
    assert image.ResolutionLevels == len(optimal_image_pyramid(volume.shape, 2))
    assert image.TimePoints == 2
    assert image.Channels == 2
    assert image.shape[-3:] == (n_z, n_y, n_x)
    np.testing.assert_array_equal(np.asarray(image[0, 0])[:n_z, :n_y, :n_x], volume)
