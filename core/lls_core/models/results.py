from __future__ import annotations
from itertools import groupby
from pathlib import Path

from typing import Iterable, Optional, Tuple, Union, cast, TYPE_CHECKING, overload
from typing_extensions import Generic, TypeVar, TypeAlias
from pydantic.v1 import BaseModel, NonNegativeInt, Field
from lls_core.types import ArrayLike, is_arraylike
from lls_core.utils import make_filename_suffix
from lls_core.writers import Writer
from pandas import DataFrame
from lls_core.workflow import RawWorkflowOutput

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from lls_core.models.lattice_data import LatticeData

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

    def roi_previews(self) -> Iterable[ArrayLike]:
        """
        Extracts a single 3D image for each ROI
        """
        import numpy as np
        def _preview(slices: Iterable[ProcessedSlice[ArrayLike]]) -> ArrayLike:
            for slice in slices:
                return slice.data
            raise Exception("This ROI has no images. This shouldn't be possible")

        for roi_index, slices in groupby(self.slices, key=lambda slice: slice.roi_index):
            yield _preview(slices)

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

class ProcessedWorkflowOutput(BaseModel, arbitrary_types_allowed=True):
    """
    Result class for one single workflow output, after it has been processed.
    """

    #: Index of this output from the workflow function. For example, if you `return a, b, c` in your final workflow step,
    #: there will be 3 `ProcessedWorkflowOutput` instances created, one with `index=0`, `index=1` and `index=2`
    index: int

    #: Index of region of interest that produced this
    roi_index: Optional[int]

    #: A processed output from the workflow. Either the path to a saved image, or a `DataFrame` capturing some other metadata.
    data: Union[Path, DataFrame]

    #: Reference to the original settings that created this
    lattice_data: LatticeData

    def save(self) -> Path:
        """
        Puts this artifact on disk by saving any `DataFrame` to CSV, and returning the path to the image or CSV
        """
        from pandas import Series

        if isinstance(self.data, DataFrame):
            path: Path = self.lattice_data.make_filepath_df(
                make_filename_suffix(
                    roi_index=self.roi_index,
                    prefix=f"_output_{self.index}"
                ),
                self.data
            )
            result = self.data.apply(Series.explode)
            result.to_csv(str(path))
            return path
        else:
            return self.data

MaybeTupleRawWorkflowOutput: TypeAlias = Union[Tuple[RawWorkflowOutput], RawWorkflowOutput]
class WorkflowSlices(ProcessedSlices[MaybeTupleRawWorkflowOutput]):
    """
    The counterpart of `ImageSlices`, but for workflow outputs.
    This is needed because workflows have vastly different outputs that may include regular
    Python types rather than only image slices.
    """

    # This re-definition of the type is helpful for `mkdocs`
    slices: Iterable[ProcessedSlice[MaybeTupleRawWorkflowOutput]] = Field(description="Iterable of raw workflow results, the exact nature of which is determined by the author of the workflow. Not typically useful directly, and using he result of `.process()` is recommended instead.")

    def process(self) -> Iterable[ProcessedWorkflowOutput]:
        """
        Incrementally processes the workflow outputs, and returns both image paths and data frames of the outputs,
        for image slices and dict/list outputs respectively
        """
        import pandas as pd
        from lls_core.models.lattice_data import LatticeData
        ProcessedWorkflowOutput.update_forward_refs(LatticeData=LatticeData)

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

            for i, element in enumerate(values):
                if isinstance(element, Writer):
                    element.close()
                    for file in element.written_files:
                        yield ProcessedWorkflowOutput(index=i, roi_index=roi, data=file, lattice_data=self.lattice_data)
                else:
                    yield ProcessedWorkflowOutput(index=i, roi_index=roi, data=pd.DataFrame(element), lattice_data=self.lattice_data)

    def roi_previews(self) -> Iterable[NDArray]:
        """
        Extracts a single 3D image for each ROI
        """
        import numpy as np
        def _preview(slices: Iterable[ProcessedSlice[MaybeTupleRawWorkflowOutput]]) -> NDArray:
            for slice in slices:
                for value in slice.as_tuple():
                    if is_arraylike(value):
                        return np.asarray(value)
            raise Exception("This ROI has no images. This shouldn't be possible")

        for roi_index, slices in groupby(self.slices, key=lambda slice: slice.roi_index):
            yield _preview(slices)

    def save(self) -> Iterable[Path]:
        """
        Processes all workflow outputs and saves them to disk.
        Images are saved in the format specified in the `LatticeData`, while
        other data types are saved as a CSV.
        **Remember to call `list()` on this to exhaust the generator and run the computation**.
        """
        for result in self.process():
            yield result.save()
