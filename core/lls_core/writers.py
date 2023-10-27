from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import groupby
from typing import TYPE_CHECKING, Iterator, List, Optional

from lls_core.types import ArrayLike

from pydantic import NonNegativeInt

from lls_core.utils import make_filename_prefix
RoiIndex = Optional[NonNegativeInt]

if TYPE_CHECKING:
    from lls_core.models.lattice_data import LatticeData
    import npy2bdv
    from lls_core.models.results import ProcessedSlice


@dataclass
class Writer(ABC):
    lattice: LatticeData
    roi_index: RoiIndex

    @abstractmethod
    def get_filepath(self) -> str:
        pass
    
    @abstractmethod
    def write_slice(self, slice: ProcessedSlice[ArrayLike]):
        pass

    def close(self):
        pass

@dataclass
class BdvWriter(Writer):
    bdv_writer: npy2bdv.BdvWriter = field(init=False)

    def get_filepath(self) -> str:
        return str(self.lattice.make_filepath(make_filename_prefix(roi_index=self.roi_index)))

    def __post_init__(self):
        import npy2bdv
        self.bdv_writer = npy2bdv.BdvWriter(
            filename=self.get_filepath(),
            compression='gzip',
            nchannels=len(self.lattice.channel_range),
            subsamp=((1, 1, 1), (1, 2, 2), (2, 4, 4)),
            overwrite=False
        )

    def write_slice(self, slice: ProcessedSlice[ArrayLike]):
        import numpy as np
        self.bdv_writer.append_view(
            np.array(slice.data),
            time=slice.time,
            channel=slice.channel,
            voxel_size_xyz=(self.lattice.dx, self.lattice.dy, self.lattice.new_dz),
            voxel_units='um'
        )

    def close(self):
        self.bdv_writer.write_xml()
        self.bdv_writer.close()

@dataclass
class TiffWriter(Writer):
    pending_slices: list = field(default_factory=list, init=False)
    time: Optional[NonNegativeInt] = None

    def get_filepath(self) -> str:
        return str(self.lattice.make_filepath(make_filename_prefix(roi_index=self.roi_index)))

    def __post_init__(self):
        self.pending_slices = []

    def write_slice(self, slice: ProcessedSlice[ArrayLike]):
        import numpy as np
        import tifffile
        if slice.time != self.time and len(self.pending_slices) > 0:
            # Write the old timepoint once we 
            first_result = self.pending_slices[0]
            images_array = np.swapaxes(np.expand_dims([result.data for result in self.pending_slices], axis=0), 1, 2)
            tifffile.imwrite(
                str(self.lattice.make_filepath(make_filename_prefix(channel=first_result.channel, time=slice.time, roi_index=slice.roi))),
                data = images_array,
                bigtiff=True,
                resolution=(1./self.lattice.dx, 1./self.lattice.dy, "MICROMETER"),
                metadata={'spacing': self.lattice.new_dz, 'unit': 'um', 'axes': 'TZCYX'},
                imagej=True
            )

            # Reinitialise
            self.pending_slices = []

        self.time = slice.time
        self.pending_slices.append(slice)
