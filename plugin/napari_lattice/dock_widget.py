from __future__ import annotations

import logging
from textwrap import dedent
from typing import TYPE_CHECKING
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
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTabWidget
from napari_lattice.parent_connect import ParentConnect

if TYPE_CHECKING:
    from typing import Iterable
    from napari_lattice.fields import NapariFieldGroup
    from lls_core.types import ArrayLike

# Enable Logging
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

    def _make_model(self, validate: bool = True) -> LatticeData:
        from rich import print
        from sys import stdout

        deskew_args = self.LlszMenu.WidgetContainer.deskew_fields._get_kwargs()
        output_args = self.LlszMenu.WidgetContainer.output_fields._make_model(validate=False)
        params = LatticeData.make(
            validate=validate, 

            # Deskew
            input_image=deskew_args["data"],
            angle=deskew_args["angle"],
            physical_pixel_sizes=deskew_args["physical_pixel_sizes"],
            skew=deskew_args["skew"],

            # Output
            channel_range=output_args.channel_range,
            time_range=output_args.time_range,
            save_dir=output_args.save_dir,
            save_name=output_args.save_name or deskew_args["save_name"],
            save_type=output_args.save_type,
            save_suffix=output_args.save_suffix,
            
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
        heading1 = field(dedent("""
        <div>
        Specify deskewing parameters and image layers in Tab 1.&nbsp; 
        Additional analysis parameters can be configured in the other tabs.&nbsp;
        When you are ready to save,&nbsp;go to Tab 5.&nbsp;
        Output to specify the output directory.&nbsp;
        For more information,&nbsp;<a href="https://github.com/BioimageAnalysisCoreWEHI/napari_lattice/wiki">please refer to the documentation here</a>.
        </div>
        """.strip()), widget_type="Label")

        def __post_init__(self):
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QLabel, QLayout

            if isinstance(self._widget._layout, QLayout):
                self._widget._layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            if isinstance(self.heading1.native, QLabel):
                self.heading1.native.setWordWrap(True)

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
                    # Connect event handlers
                    for subfield_name in dir(field):
                        subfield = getattr(field, subfield_name)
                        if isinstance(subfield, ParentConnect):
                            subfield.resolve(self, field, subfield_name)
                    # Trigger validation of default data
                    field._validate()

            # Using vfields here seems to prevent https://github.com/hanjinliu/magic-class/issues/110
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
    def preview(self, header: str, time: int, channel: int):
        from pathlib import Path

        # We only need to process one time point for the preview, 
        # so we made a copy using a subset of the times
        lattice = self._make_model(validate=False).copy_validate(update=dict(
            time_range = range(time, time+1),
            channel_range = range(channel, channel+1),
            # Patch in a placeholder for the save dir because previewing doesn't use it
            # TODO: use a more elegant solution such as making the "saveable" lattice
            # a child class which more validations
            save_dir = Path.home()
        ))

        scale = (
            lattice.new_dz,
            lattice.dy,
            lattice.dx
        )
        previews: Iterable[ArrayLike]

        # We extract the first available image to use as a preview
        # This works differently for workflows and non-workflows
        if lattice.workflow is None:
            previews = lattice.process().roi_previews()
        else:
            previews = lattice.process_workflow().roi_previews()

        for preview in previews:
            self.parent_viewer.add_image(preview, scale=scale, name="Napari Lattice Preview")
            max_z = np.argmax(np.sum(preview, axis=(1, 2)))
            self.parent_viewer.dims.set_current_step(0, max_z)


    @set_design(text="Save")
    def save(self):
        from napari.utils.notifications import show_info
        lattice = self._make_model()
        lattice.save()
        show_info(f"Deskewing successfuly completed. Results are located in {lattice.save_dir}")

    def _get_fields(self) -> Iterable[NapariFieldGroup]:
        """Yields all the child Field classes which inherit from NapariFieldGroup"""
        container = self.LlszMenu.WidgetContainer
        yield from set(container.__magicclass_children__)
