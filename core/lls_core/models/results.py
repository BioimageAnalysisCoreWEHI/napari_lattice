from __future__ import annotations
from itertools import groupby
from pathlib import Path

from typing import Iterable, Optional, Tuple, Union, cast, TYPE_CHECKING, overload
from typing_extensions import Generic, TypeVar
from pydantic.v1 import BaseModel, NonNegativeInt, Field
from lls_core.types import ArrayLike, is_arraylike
from lls_core.utils import make_filename_suffix
from lls_core.writers import RoiIndex, Writer
from pandas import DataFrame
from lls_core.workflow import RawWorkflowOutput

if TYPE_CHECKING:
    from lls_core.models.lattice_data import LatticeData
    from numpy.typing import NDArray

T = TypeVar("T")
S = TypeVar("S")
R = TypeVar("R")
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
    
    @overload
    def as_tuple(self: ProcessedSlice[Tuple[R]]) -> Tuple[R]:
        ...
    @overload
    def as_tuple(self: ProcessedSlice[T]) -> Tuple[T]:
        ...
    def as_tuple(self):
        """
        Converts the results to a tuple if they weren't already
        """
        return self.data if isinstance(self.data, (tuple, list)) else (self.data,)

class ProcessedSlices(BaseModel, Generic[T], arbitrary_types_allowed=True):
    """
    A generic parent class for holding deskewing outputs.
    This will never be instantiated directly.
    Refer to the concrete child classes for more detail.
    """
    slices: Iterable[ProcessedSlice[T]] = Field(description="Iterable of result slices. Note that this is a finite iterator that can only be iterated once")
    lattice_data: LatticeData = Field(description='The "parent" LatticeData that was used to create this result')


ImageSlice = ProcessedSlice[ArrayLike]
class ImageSlices(ProcessedSlices[ArrayLike]):
    """
    A collection of image slices, which is the main output from deskewing.
    This holds an iterable of output image slices before they are saved to disk,
    and provides a `save_image()` method for this purpose.
    """

    # This re-definition of the type is helpful for `mkdocs`
    slices: Iterable[ProcessedSlice[ArrayLike]] = Field(description="Iterable of result slices. For a given slice, you can access the image data through the `slice.data` property, which is a numpy-like array.")

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

ProcessedWorkflowOutput = Union[
    # A path indicates a saved file
    Path,
    DataFrame
]
"""
The result of a workflow. If this is a `Path`, then it is the path to an image saved to disk.
If a `DataFrame`, then it contains non-image data returned by your workflow.
"""

class WorkflowSlices(ProcessedSlices[Union[Tuple[RawWorkflowOutput], RawWorkflowOutput]]):
    """
    The counterpart of `ImageSlices`, but for workflow outputs.
    This is needed because workflows have vastly different outputs that may include regular
    Python types rather than only image slices.
    """

    # This re-definition of the type is helpful for `mkdocs`
    slices: Iterable[ProcessedSlice[Union[Tuple[RawWorkflowOutput], RawWorkflowOutput]]] = Field(description="Iterable of raw workflow results, the exact nature of which is determined by the author of the workflow. Not typically useful directly, and using he result of `.process()` is recommended instead.")

    def process(self) -> Iterable[Tuple[RoiIndex, ProcessedWorkflowOutput]]:
        """
        Incrementally processes the workflow outputs, and returns both image paths and data frames of the outputs,
        for image slices and dict/list outputs respectively
        """
        import pandas as pd

        # Handle each ROI separately
        for roi, roi_results in groupby(self.slices, key=lambda it: it.roi_index):
            values: list[Union[Writer, list]] = []
            for result in roi_results:
                # If the user didn't return a tuple, put it into one
                for i, element in enumerate(result.as_tuple()):
                    # If the element is array like, we assume it's an image to write to disk
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
                        # Otherwise, we assume it's one row to be added to a data frame
                        if len(values) <= i:
                            values.append([])

                        rows = cast(list, values[i])

                        if isinstance(element, dict):
                            # If the row is a dict, it has column names
                            element = {"time": f"T{result.time_index}", "channel": f"C{result.channel_index}", **element}
                        elif isinstance(element, Iterable):
                            # If the row is a list, it has no column names
                            # We add the channel and time 
                            element = [f"T{result.time_index}", f"C{result.channel_index}", *element]
                        else:
                            # If the row is just a value, we turn that value into a single column of the data frame
                            element = [f"T{result.time_index}", f"C{result.channel_index}", element]

                        rows.append(element)

            for element in values:
                if isinstance(element, Writer):
                    element.close()
                    for file in element.written_files:
                        yield roi, file
                else:
                    yield roi, pd.DataFrame(element)

    def extract_preview(self) -> NDArray:
        """
        Extracts a single 3D image for previewing purposes
        """
        import numpy as np
        for slice in self.slices:
            for value in slice.as_tuple():
                if is_arraylike(value):
                    return np.asarray(value)
        raise Exception("No image was returned from this workflow")

    def save(self) -> Iterable[Path]:
        """
        Processes all workflow outputs and saves them to disk.
        Images are saved in the format specified in the `LatticeData`, while
        other data types are saved as a CSV.
        """
        from pandas import DataFrame, Series
        for i, (roi, result) in enumerate(self.process()):
            if isinstance(result, DataFrame):
                path = self.lattice_data.make_filepath_df(make_filename_suffix(roi_index=roi, prefix=f"_output_{i}"), result)
                result = result.apply(Series.explode)
                result.to_csv(str(path))
                yield path
            else:
                yield result
