from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from lls_core.types import ArrayLike

from pydantic.v1 import NonNegativeInt

from lls_core.utils import make_filename_suffix
RoiIndex = Optional[NonNegativeInt]

if TYPE_CHECKING:
    from lls_core.models.lattice_data import LatticeData
    import npy2bdv
    from lls_core.models.results import ProcessedSlice, ImageSlice
    from pathlib import Path


@dataclass
class Writer(ABC):
    """
    A writer is an abstraction over the logic used to write image slices to disk.
    `Writer`s need to work incrementally, in order that we don't need the entire multidimensional
    image in memory at the same time.
    """
    lattice: LatticeData
    roi_index: RoiIndex
    written_files: List[Path] = field(default_factory=list, init=False)

    @abstractmethod
    def write_slice(self, slice: ProcessedSlice[ArrayLike]):
        """
        Writes a 3D image slice
        """
        pass

    def close(self):
        """
        Called when no more image slices are available, and the writer should finalise its output files
        """
        pass

@dataclass
class BdvWriter(Writer):
    """
    A writer for for Fiji BigDataViewer output format
    """
    bdv_writer: npy2bdv.BdvWriter = field(init=False)

    def __post_init__(self):
        import npy2bdv
        suffix = f"_{make_filename_suffix(roi_index=str(self.roi_index))}" if self.roi_index is not None else ""
        path = self.lattice.make_filepath(suffix)
        self.bdv_writer = npy2bdv.BdvWriter(
            filename=str(path),
            compression='gzip',
            nchannels=len(self.lattice.channel_range),
            subsamp=((1, 1, 1), (1, 2, 2), (2, 4, 4)),
            overwrite=False
        )
        self.written_files.append(path)

    def write_slice(self, slice: ProcessedSlice[ArrayLike]):
        import numpy as np
        self.bdv_writer.append_view(
            np.array(slice.data),
            # We need to use the indices here to ensure they start from 0 and 
            # are contiguous
            time=slice.time_index,
            channel=slice.channel_index,
            voxel_size_xyz=(self.lattice.dx, self.lattice.dy, self.lattice.new_dz),
            voxel_units='um'
        )

    def close(self):
        self.bdv_writer.write_xml()
        self.bdv_writer.close()

@dataclass
class TiffWriter(Writer):
    """
    A writer for for TIFF output format
    """
    pending_slices: List[ImageSlice] = field(default_factory=list, init=False)
    time: Optional[NonNegativeInt] = None

    def __post_init__(self):
        self.pending_slices = []
    
    def flush(self):
        "Write out all pending slices"
        import numpy as np
        import tifffile
        if len(self.pending_slices) > 0:
            first_result = self.pending_slices[0]
            images_array = np.swapaxes(
                np.expand_dims([result.data for result in self.pending_slices], axis=0),
                1, 2
            ).astype("uint16")
            # ImageJ TIFF can only handle 16-bit uints, not 32
            path = self.lattice.make_filepath(
                make_filename_suffix(
                    channel=first_result.channel,
                    time=first_result.time,
                    roi_index=first_result.roi_index
                )
            )
            tifffile.imwrite(
                str(path),
                data = images_array,
                bigtiff=True,
                resolution=(1./self.lattice.dx, 1./self.lattice.dy, "MICROMETER"),
                metadata={'spacing': self.lattice.new_dz, 'unit': 'um', 'axes': 'TZCYX'},
                imagej=True
            )
            self.written_files.append(path)

            # Reinitialise
            self.pending_slices = []

    def write_slice(self, slice: ProcessedSlice[ArrayLike]):
        if slice.time != self.time:
            self.flush()

        self.time = slice.time
        self.pending_slices.append(slice)

    def close(self):
        self.flush()
