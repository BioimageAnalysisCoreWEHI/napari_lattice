from __future__ import annotations

from importlib_resources import as_file
from napari_lattice.dock_widget import LLSZWidget
from typing import Callable, TYPE_CHECKING
from magicclass.testing import check_function_gui_buildable, FunctionGuiTester
from magicclass import MagicTemplate
from magicclass.widgets import Widget
from magicclass.utils import thread_worker
from magicclass._gui._gui_modes import ErrorMode
import pytest
from lls_core.sample import resources
from bioio import BioImage
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
    #"RBC_tiny.czi", Removing as it adds ~10 min to test being a bigger file. Other files are sufficient for this test
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
        yield BioImage(image_path, )

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

def test_dock_widget(make_napari_viewer: Callable[[], Viewer], image_data: BioImage):
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

        # thread_worker methods run async on a QThread when called via the GUI;
        # under test there is no spinning event loop, so force blocking mode so
        # the worker bodies and their returned/yielded callbacks run synchronously.
        with thread_worker.blocking_mode():
            # Test previewing
            tester = FunctionGuiTester(ui.preview)
            tester.call("", 0, 0)

            # Add the save path which shouldn't be needed for previewing
            ui.LlszMenu.WidgetContainer.output_fields.save_path.value = tmpdir

            # Test saving
            tester = FunctionGuiTester(ui.save)
            tester.call()


def test_get_kwargs_caches_reader(make_napari_viewer: Callable[[], Viewer], monkeypatch):
    """
    `_get_kwargs` must reuse the cached reader output when only deskew scalars
    change, and re-run the reader when an image-side input changes. This is the
    validation-cost fix: the ~300 ms image concat should not run on every
    angle/skew tweak.
    """
    import napari_lattice.fields as fmod

    viewer = make_napari_viewer()
    with as_file(resources / "RBC_tiny.czi") as image_path:
        image_data = BioImage(image_path)
        viewer.add_image(image_data.xarray_dask_data)

        ui = LLSZWidget()
        set_debug(ui)
        viewer.window.add_dock_widget(ui)

        fields = ui.LlszMenu.WidgetContainer.deskew_fields
        fields.img_layer.value = list(viewer.layers)
        fields.dimension_order.value = image_data.dims.order
        fields.pixel_sizes_source.value = PixelSizeSource.Manual

        # Count reader invocations without changing behaviour.
        calls = {"n": 0}
        real = fmod.lattice_params_from_napari
        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)
        monkeypatch.setattr(fmod, "lattice_params_from_napari", counting)

        # Prime the cache, then measure deltas from here.
        fields._get_kwargs()
        calls["n"] = 0

        # (1) Identical inputs -> cache hit, reader not called.
        fields._get_kwargs()
        assert calls["n"] == 0

        # (2) Changing a deskew scalar must NOT invalidate the cache.
        fields.angle.value = 32.0
        fields._get_kwargs()
        assert calls["n"] == 0

        # (3) Changing an image-side input MUST invalidate the cache.
        pv = fields.pixel_sizes.value
        fields.pixel_sizes.value = (pv[0] + 1.0, pv[1], pv[2])
        fields._get_kwargs()
        assert calls["n"] >= 1


def test_quick_deskew_toggle_restores_raw_scale(make_napari_viewer: Callable[[], Viewer]):
    """
    Toggling Quick Deskew OFF must restore the raw-view scale. Regression for the
    fix wiring `_rescale_image` to the `quick_deskew` signal — previously the
    layer was left at the deskewed z-spacing after turning Quick Deskew off.
    """
    import numpy as np

    viewer = make_napari_viewer()
    with as_file(resources / "RBC_tiny.czi") as image_path:
        image_data = BioImage(image_path)
        viewer.add_image(image_data.xarray_dask_data)

        ui = LLSZWidget()
        set_debug(ui)
        viewer.window.add_dock_widget(ui)

        fields = ui.LlszMenu.WidgetContainer.deskew_fields
        fields.img_layer.value = list(viewer.layers)
        fields.dimension_order.value = image_data.dims.order
        fields.pixel_sizes_source.value = PixelSizeSource.Manual
        fields.pixel_sizes.value = (0.15, 0.15, 0.3)  # X, Y, Z

        layer = list(viewer.layers)[0]
        raw = tuple(np.asarray(layer.scale)[-3:])          # QD off after setup

        fields.quick_deskew.value = True
        deskewed = tuple(np.asarray(layer.scale)[-3:])     # deskewed z-spacing

        fields.quick_deskew.value = False
        restored = tuple(np.asarray(layer.scale)[-3:])     # must be raw again

        assert deskewed != raw, "enabling Quick Deskew should change the scale"
        assert np.allclose(restored, raw), "disabling Quick Deskew must restore raw scale"


def test_check_buildable():
    ui = LLSZWidget()
    set_debug(ui)
    check_function_gui_buildable(ui)


def test_parallel_roi_save_off_main_thread():
    """
    Regression: calling LatticeData.save() from off the main thread (as the GUI
    now does via thread_worker) must not break the parallel-ROI process/thread
    pools. Engine correctness itself is covered in core/tests/test_parallel_processing.py.

    Construction mirrors core/tests/test_parallel_processing.py::_make_lattice.
    A 2-ROI crop with process_parallel=2 forces the parallel-ROI save path
    (_use_parallel_roi_processing: >1 ROI and >1 worker).
    """
    import os
    import numpy as np
    from xarray import DataArray
    from magicclass import magicclass
    from magicclass.utils import thread_worker
    from lls_core.models.lattice_data import LatticeData
    from lls_core.models.crop import CropParams

    def _roi(y0, x0, y1, x1):
        return [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]

    raw = np.zeros((30, 90, 90), dtype=np.uint16)
    rois = [_roi(0, 0, 30, 30), _roi(0, 30, 30, 60)]  # 2 ROIs

    with TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(1, 1, 1),
            save_name="test",
            save_dir=tmpdir,
            save_type="tiff",
            crop=CropParams(roi_list=rois, z_range=(0, 20)),
            process_parallel=2,
        )

        # Sanity: this really is the parallel path, not the serial fallback.
        assert lattice._use_parallel_roi_processing()

        @magicclass
        class _Runner:
            @thread_worker
            def run(self):
                lattice.save()

        runner = _Runner()
        with thread_worker.blocking_mode():
            runner.run()  # runs the worker body synchronously to completion

        produced = list(os.scandir(tmpdir))
        assert produced, "parallel ROI save produced no output files"


def test_custom_workflow_is_handed_over_as_a_path(tmp_path):
    """
    The output metadata sidecar records the workflow by its source path, and a `Workflow`
    object cannot say where it was read from. So these fields must hand `LatticeData` the
    path and let its validators do the loading.

    Regression: loading the workflow here instead threw the path away, and every run
    configured through the GUI wrote `"workflow": null` into its sidecar.
    """
    from napari_lattice.fields import WorkflowSource

    workflow_path = tmp_path / "flow.yml"
    workflow_path.write_text("""!!python/object:napari_workflows._workflow.Workflow
_tasks:
  blurred: !!python/tuple
  - !!python/name:pyclesperanto_prototype.gaussian_blur ''
  - deskewed_image
  - null
  - 1
  - 1
  - 1
""")

    ui = LLSZWidget()
    set_debug(ui)
    fields = ui.LlszMenu.WidgetContainer.workflow_fields
    fields.fields_enabled.value = True
    fields.workflow_source.value = WorkflowSource.CustomPath
    fields.workflow_path.value = workflow_path

    assert fields._make_model() == workflow_path
