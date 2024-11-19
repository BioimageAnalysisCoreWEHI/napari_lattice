from __future__ import annotations

from importlib_resources import as_file
from napari_lattice.dock_widget import LLSZWidget
from typing import Callable, TYPE_CHECKING
from magicclass.testing import check_function_gui_buildable, FunctionGuiTester
from magicclass import MagicTemplate
from magicclass.widgets import Widget
from magicclass._gui._gui_modes import ErrorMode
import pytest
from lls_core.sample import resources
from aicsimageio.aics_image import AICSImage
from napari_lattice.fields import PixelSizeSource
from tempfile import TemporaryDirectory

if TYPE_CHECKING:
    from napari import Viewer

# Test if the widget can be created

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# Commenting this out as github CI is fixed
# @pytest.mark.skip(reason="GUI tests currently fail in github CI, unclear why")
# When testing locally, need pytest-qt

@pytest.fixture(params=[
    "RBC_tiny.czi",
    "LLS7_t1_ch1.czi",
    "LLS7_t1_ch3.czi",
    "LLS7_t2_ch1.czi",
    "LLS7_t2_ch3.czi",
])
def image_data(request: pytest.FixtureRequest):
    """
    Fixture function that yields test images as file paths
    """
    with as_file(resources / request.param) as image_path:
        yield AICSImage(image_path, )

def set_debug(cls: MagicTemplate):
    """
    Recursively disables GUI error handling, so that this works with pytest
    """
    def _handler(e: Exception, parent: Widget):
        raise e
    ErrorMode.get_handler = lambda self: _handler
    cls._error_mode = ErrorMode.stderr
    for child in cls.__magicclass_children__:
        set_debug(child)

def test_dock_widget(make_napari_viewer: Callable[[], Viewer], image_data: AICSImage):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # Check if an image can be added as a layer
    viewer.add_image(image_data.xarray_dask_data)

    # Test if napari-lattice widget can be created in napari
    ui = LLSZWidget()
    set_debug(ui)
    viewer.window.add_dock_widget(ui)

    # Set the input parameters and execute the processing
    with TemporaryDirectory() as tmpdir:
        # Specify values for all the required GUI fields
        fields = ui.LlszMenu.WidgetContainer.deskew_fields
        # TODO: refactor this logic into a `lattice_params_from_aics` method
        fields.img_layer.value = list(viewer.layers)
        fields.dimension_order.value = image_data.dims.order
        fields.pixel_sizes_source.value = PixelSizeSource.Manual

        # Test previewing
        tester = FunctionGuiTester(ui.preview)
        tester.call("", 0, 0)

        # Add the save path which shouldn't be needed for previewing
        ui.LlszMenu.WidgetContainer.output_fields.save_path.value = tmpdir
        
        # Test saving
        tester = FunctionGuiTester(ui.save)
        tester.call()


def test_check_buildable():
    ui = LLSZWidget()
    set_debug(ui)
    check_function_gui_buildable(ui)
