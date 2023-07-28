from __future__ import annotations
from napari_lattice._dock_widget import _napari_lattice_widget_wrapper
import numpy as np
from typing import Callable, TYPE_CHECKING
from magicclass.testing import check_function_gui_buildable, FunctionGuiTester
from napari.layers import Image
from magicclass import MagicTemplate
from magicclass.widgets import Widget
from magicclass._gui._gui_modes import ErrorMode
import pytest

if TYPE_CHECKING:
    from napari import Viewer

# Test if the widget can be created

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# Commenting this out as github CI is fixed
# @pytest.mark.skip(reason="GUI tests currently fail in github CI, unclear why")
# When testing locally, need pytest-qt

@pytest.fixture
def debug_widget():
    """
    Recursively disables GUI error handling, so that this works with pytest.
    This is a massive hack while we wait for https://github.com/hanjinliu/magic-class/issues/109
    """
    def _handler(e: Exception, parent: Widget):
        raise e
    ErrorMode.get_handler = lambda self: _handler
    old_error_mode = cls._error_mode
    cls._error_mode = ErrorMode.stderr
    for child in cls.__magicclass_children__:
        set_debug(child)

def test_dock_widget(make_napari_viewer: Callable[[], Viewer]):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # Check if an image can be added as a layer
    viewer.add_image(np.random.random((100, 100)))

    # Test if napari-lattice widget can be created in napari
    gui = _napari_lattice_widget_wrapper()
    viewer.window.add_dock_widget(gui)

def test_check_buildable():
    widget = _napari_lattice_widget_wrapper()
    check_function_gui_buildable(widget)

def test_plugin_initialize(make_napari_viewer: Callable[[], Viewer]):
    ui = _napari_lattice_widget_wrapper()
    viewer = make_napari_viewer()
    viewer.window.add_dock_widget(ui)
    image = Image(np.random.random((100, 100, 100, 100)))
    set_debug(ui)
    tester = FunctionGuiTester(ui.LlszMenu.Choose_Image_Layer)
    tester.call(img_layer=image, last_dimension_channel="time")
