# Enable Logging
import logging

import numpy as np
from lls_core.models.lattice_data import LatticeData
from magicclass import MagicTemplate, field, magicclass, set_options, vfield
from magicclass.wrappers import set_design
from napari_lattice.fields import (
    CroppingFields,
    DeconvolutionFields,
    DeskewFields,
    OutputFields,
    WorkflowFields,
)
from napari_lattice.icons import GREY
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QTabWidget

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@magicclass(widget_type="split")
class LLSZWidget(MagicTemplate):
    def __post_init__(self):
        # aligning collapsible widgets at the top instead of having them centered vertically
        self._widget._layout.setAlignment(Qt.AlignTop)

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
        from rich import print
        from sys import stdout

        deskew_args = self.LlszMenu.WidgetContainer.deskew_fields._get_kwargs()
        output_args = self.LlszMenu.WidgetContainer.output_fields._make_model()
        params = LatticeData(
            input_image=deskew_args["data"],
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
        # Log the lattice
        print(params, file=stdout)
        return params

    @magicclass(widget_type="split")
    class LlszMenu(MagicTemplate):

        main_heading = field("<h3>Napari Lattice: Visualization & Analysis</h3>", widget_type="Label")
        heading1 = field("Select the tabs one at a time to specify analysis parameters. Tabs 1 and 5 are mandatory.", widget_type="Label")

        # Tabbed Widget container to house all the widgets
        @magicclass(widget_type="tabbed", name="Functions", labels=False)
        class WidgetContainer(MagicTemplate):

            def __post_init__(self):
                tab_widget: QTabWidget= self._widget._tab_widget
                # Manually set the tab labels, because by default magicgui uses the widget names, but setting
                # the names to human readable text makes them difficult to access via self
                for i, label in enumerate(["1. Deskew", "2. Deconvolution", "3. Crop", "4. Workflow", "5. Output"]):
                    tab_widget.setTabText(i, label)
                for field in [self.deskew_fields, self.deconv_fields, self.cropping_fields, self.workflow_fields, self.output_fields]:
                    field._validate()

            deskew_fields = vfield(DeskewFields)
            deconv_fields = vfield(DeconvolutionFields)
            cropping_fields = vfield(CroppingFields)
            workflow_fields = vfield(WorkflowFields)
            output_fields = vfield(OutputFields)

    @set_options(header=dict(widget_type="Label", label="<h3>Preview Deskew</h3>"),
                time=dict(label="Time:", max=2**15),
                channel=dict(label="Channel:"),
                call_button="Preview"
                )
    @set_design(text="Preview")
    def preview(self, header:str, time: int, channel: int):
        # We only need to process one time point for the preview, 
        # so we made a copy using a subset of the times
        lattice = self._make_model().copy_validate(update=dict(
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
        from napari.utils.notifications import show_info
        lattice = self._make_model()
        lattice.process().save_image()
        show_info(f"Deskewing successfuly completed. Results are located in {lattice.save_dir}")
