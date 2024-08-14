from __future__ import annotations
from itertools import groupby
from pathlib import Path

from typing import Iterable, Optional, Tuple, Union, cast, TYPE_CHECKING
from typing_extensions import Generic, TypeVar
from pydantic import BaseModel, NonNegativeInt
from lls_core.types import ArrayLike, is_arraylike
from lls_core.utils import make_filename_suffix
from lls_core.writers import RoiIndex, Writer
from pandas import DataFrame, Series

if TYPE_CHECKING:
    from lls_core.models.lattice_data import LatticeData

T = TypeVar("T")
S = TypeVar("S")
class ProcessedSlice(BaseModel, Generic[T], arbitrary_types_allowed=True):
    """
    A single slice of some data that is split across multiple slices along time or channel axes
    This class is generic over T, the type of data that is sliced.
    """
    data: T
    time_index: NonNegativeInt
    time: NonNegativeInt
    channel_index: NonNegativeInt
    channel: NonNegativeInt
    roi_index: Optional[NonNegativeInt] = None

    def copy_with_data(self, data: S) -> ProcessedSlice[S]:
        """
        Return a modified version of this with new inner data
        """
        from typing_extensions import cast
        return cast(
            ProcessedSlice[S],
            self.copy(update={
                "data": data
            })
        ) 

class ProcessedSlices(BaseModel, Generic[T], arbitrary_types_allowed=True):
    """
    A generic parent class for holding deskewing outputs.
    This will never be instantiated directly.
    Refer to the concrete child classes for more detail.
    """
    #: Iterable of result slices.
    #: Note that this is a finite iterator that can only be iterated once
    slices: Iterable[ProcessedSlice[T]]

    #: The "parent" LatticeData that was used to create this result
    lattice_data: LatticeData

ImageSlice = ProcessedSlice[ArrayLike]
class ImageSlices(ProcessedSlices[ArrayLike]):
    """
    A collection of image slices, which is the main output from deskewing.
    This holds an iterable of output image slices before they are saved to disk,
    and provides a `save_image()` method for this purpose.
    """
    def save_image(self):
        """
        Saves result slices to disk
        """
        Writer = self.lattice_data.get_writer()
        for roi, roi_results in groupby(self.slices, key=lambda it: it.roi_index):
            writer = Writer(self.lattice_data, roi_index=roi)
            for slice in roi_results:
                writer.write_slice(slice)
            writer.close()


RawWorkflowOutput = Union[
    ArrayLike,
    dict,
    list
]
ProcessedWorkflowOutput = Union[
    # A path indicates a saved file
    Path,
    DataFrame
]

class WorkflowSlices(ProcessedSlices[Tuple[RawWorkflowOutput]]):
    """
    The counterpart of `ImageSlices`, but for workflow outputs.
    This is needed because workflows have vastly different outputs that may include regular
    Python types rather than only image slices.
    """
    def process(self) -> Iterable[Tuple[RoiIndex, ProcessedWorkflowOutput]]:
        """
        Incrementally processes the workflow outputs, and returns both image paths and data frames of the outputs,
        for image slices and dict/list outputs respectively
        """
        import pandas as pd

        # Handle each ROI separately
        for roi, roi_results in groupby(self.slices, key=lambda it: it.roi_index):
            values: list[Writer, dict, tuple, list] = []
            for result in roi_results:
                # Ensure the data is in a tuple
                data = (result.data,) if is_arraylike(result.data) else result.data
                for i, element in enumerate(data):
                    if is_arraylike(element):
                        # Make the writer the first time only
                        if len(values) <= i:
                            values.append(self.lattice_data.get_writer()(self.lattice_data, roi_index=roi))

                        writer = cast(Writer, values[i])
                        writer.write_slice(
                            result.copy_with_data(
                                element
                            )
                        )
                    else:
                        if len(values) <= i:
                            values.append([])

                        rows = cast(list, values[i])
                        rows.append(element)

            for element in values:
                if isinstance(element, Writer):
                    element.close()
                    for file in element.written_files:
                        yield roi, file
                else:
                    yield roi, pd.DataFrame(element)

    def save(self) -> Iterable[Path]:
        """
        Processes all workflow outputs and saves them to disk.
        Images are saved in the format specified in the `LatticeData`, while
        other data types are saved as a data frame.
        """
        for roi, result in self.process():
            if isinstance(result, DataFrame):
                path = self.lattice_data.make_filepath_df(make_filename_suffix(roi_index=roi),result)
                result = result.apply(Series.explode)
                result.to_csv(str(path))
                yield path
            else:
                yield result
