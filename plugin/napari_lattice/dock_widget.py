# Enable Logging
import logging
from pathlib import Path
from typing import Union

import numpy as np
from lls_core.lattice_data import LatticeData
from lls_core.workflow import import_workflow_modules
from magicclass import MagicTemplate, field, magicclass, set_options
from magicclass.wrappers import set_design
from napari import Viewer
from napari.layers import Shapes
from napari_lattice.fields import (
    CroppingFields,
    DeconvolutionFields,
    DeskewFields,
    OutputFields,
    WorkflowFields,
)
from napari_lattice.icons import GREY
from napari_workflows import Workflow, WorkflowManager
from napari_workflows._io_yaml_v1 import load_workflow
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QTabWidget

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LlszTemplate(MagicTemplate):
    @property
    def llsz_parent(self) -> "LLSZWidget":
        return self.find_ancestor(LLSZWidget)
        
def parent_viewer(mc: MagicTemplate) -> Viewer:
    viewer = mc.parent_viewer
    if viewer is None:
        raise Exception("This function can only be used when inside of a Napari viewer")
    return mc.parent_viewer

@magicclass(widget_type="split")
class LLSZWidget(LlszTemplate):
    open_file: bool = False
    shapes_layer: Shapes

    def _check_validity(self) -> bool:
        """
        Returns True if the model is valid
        """
        try:
            self._make_model()
            return True
        except:
            return False

    def _make_model(self) -> LatticeData:
        deskew_args = self.LlszMenu.WidgetContainer.deskew_fields._get_kwargs()
        output_args = self.LlszMenu.WidgetContainer.output_fields._make_model()
        return LatticeData(
            image=deskew_args["data"],
            angle=deskew_args["angle"],
            channel_range=output_args.channel_range,
            time_range=output_args.time_range,
            save_dir=output_args.save_dir,
            # We let the user specify a prefix, but if they don't, we can use the default
            save_name=output_args.save_name or deskew_args["save_name"] ,
            save_type=output_args.save_type,
            physical_pixel_sizes=deskew_args["physical_pixel_sizes"],
            skew=deskew_args["skew"],
            workflow=self.LlszMenu.WidgetContainer.workflow_fields._make_model(),
            deconvolution=self.LlszMenu.WidgetContainer.deconv_fields._make_model(),
            crop=self.LlszMenu.WidgetContainer.cropping_fields._make_model()
        )

    @magicclass(widget_type="split")
    class LlszMenu(LlszTemplate):

        main_heading = field("<h3>Napari Lattice: Visualization & Analysis</h3>", widget_type="Label")
        heading1 = field("Drag and drop an image file onto napari.", widget_type="Label")

        # Tabbed Widget container to house all the widgets
        @magicclass(widget_type="tabbed", name="Functions", labels=False)
        class WidgetContainer(LlszTemplate):

            def __post_init__(self):
                tab_widget: QTabWidget= self._widget._tab_widget
                for i in range(5):
                    tab_widget.setTabIcon(i, QIcon(GREY))
                for field in [self.deskew_fields, self.deconv_fields, self.cropping_fields, self.workflow_fields, self.output_fields]:
                    field._validate()

            deskew_fields = DeskewFields(name = "1. Deskew")
            deconv_fields = DeconvolutionFields(name = "2. Deconvolution")
            cropping_fields = CroppingFields(name = "3. Crop")
            workflow_fields = WorkflowFields(name = "4. Workflow")
            output_fields = OutputFields(name = "5. Output")

    @set_options(header=dict(widget_type="Label", label="<h3>Preview Deskew</h3>"),
                time=dict(label="Time:", max=2**15),
                channel=dict(label="Channel:"),
                call_button="Preview"
                )
    @set_design(text="Preview")
    def preview(self, header:str, time: int, channel: int):
        # We only need to process one time point for the preview, 
        # so we made a copy using a subset of the times
        lattice = self._make_model().copy(update=dict(
            time_range = range(time, time+1),
            channel_range = range(time, time+1),
        ))

        for slice in lattice.process().slices:
            scale = (
                lattice.new_dz,
                lattice.dy,
                lattice.dx
            )
            self.parent_viewer.add_image(slice.data, scale=scale)
            max_z = np.argmax(np.sum(slice.data, axis=(1, 2)))
            self.parent_viewer.dims.set_current_step(0, max_z)

    @set_design(text="Save")
    def save(self):
        lattice = self._make_model()
        lattice.process().save_image()

def _napari_lattice_widget_wrapper() -> LLSZWidget:
    # split widget type enables a resizable widget
    #max_height = 50
    # Important to have this or napari won't recognize the classes and magicclass qidgets
    widget = LLSZWidget()
    # aligning collapsible widgets at the top instead of having them centered vertically
    widget._widget._layout.setAlignment(Qt.AlignTop)

    # widget._widget._layout.setWidgetResizable(True)
    return widget

def get_workflow(source: Union[Path, Viewer]) -> Workflow:
    """
    Gets a user defined workflow object, either from a viewer or from a file

    Args:
        source: Either the path to a workflow file, or a Napari viewer from which to extract the workflow
    """
    if isinstance(source, Viewer):
        # installs the workflow to napari
        user_workflow = WorkflowManager.install(source).workflow
        logger.info("Workflow installed")
    else:
        import_workflow_modules(source)
        user_workflow = load_workflow(str(source))

    if not isinstance(user_workflow, Workflow):
        raise Exception("Workflow file is not a napari workflow object. Check file! You can use workflow inspector if needed")

    logger.info(f"Workflow loaded: {user_workflow}")
    return user_workflow
