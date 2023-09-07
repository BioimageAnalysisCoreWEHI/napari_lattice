from pathlib import Path
from magicclass import FieldGroup, field, MagicTemplate
from magicclass.widgets import Widget, ComboBox, Label, Select
from magicclass.fields import MagicField
from typing import Any, Callable, List, Optional, Protocol, Tuple, TypeVar, Union, cast, TYPE_CHECKING
from typing_extensions import Protocol, Self
from pydantic import BaseModel, ValidationError

from strenum import StrEnum
from lls_core import DeconvolutionChoice, SaveFileType, Log_Levels, DeskewDirection
from lls_core.lattice_data import CropParams, DeconvolutionParams, DefinedPixelSizes, LatticeData, OutputParams, DeskewParams
from napari.layers import Shapes
from enum import Enum
import pyclesperanto_prototype as cle
from napari_workflows import Workflow, WorkflowManager
from napari.types import ImageData, ShapesData
from napari.utils import history
from abc import ABC
from qtpy.QtWidgets import QTabWidget

from napari_lattice.icons import RED, GREEN, GREY
from napari_lattice.reader import lattice_params_from_napari
from napari_lattice.utils import get_viewer, get_layers

from napari.layers import Image


# FieldGroups that the users interact with to input data

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def exception_to_html(e: BaseException) -> str:
    """
    Converts an exception to HTML for reporting back to the user
    """
    if isinstance(e, ValidationError):
        message = []
        for error in e.errors():
            header = ", ".join([str(it).capitalize() for it in error['loc']])
            message.append(f"<li> <b>{header}</b> {error['msg']} </li>")
        joined = '\n'.join(message)
        return f"<ul>{joined}</ul>"
    else:
        return str(e)

def get_friendly_validations(model: FieldGroup) -> str:
    """
    Generates a BaseModel, but returns validation errors in a user friendly way
    """
    try:
        model._make_model()
        return ""
    except BaseException as e:
        return exception_to_html(e)

class WorkflowSource(StrEnum):
    ActiveWorkflow = "Active Workflow"
    CustomPath = "Custom Path"

class BackgroundSource(StrEnum):
    Auto = "Automatic"
    SecondLast = "Second Last"
    Custom = "Custom"


def enable_field(parent: MagicTemplate, field: MagicField, enabled: bool = True) -> None:
    """
    Enable the widget associated with a field

    Args:
        parent: The parent magicclass that contains the field
        field: The field to enable/disable
        enabled: If False, disable the field instead of enabling it
    """
    real_field = getattr(parent, field.name)
    if not isinstance(real_field, Widget):
        raise Exception("Define your fields with field() not vfield()!")
    real_field.visible = enabled
    real_field.enabled = enabled


EnabledHandlerType = TypeVar("EnabledHandlerType")
def enable_if(fields: List[MagicField]):
    """ 
    Makes an event handler that should be used via `fields_enabled.connect(make_enabled_handler())`.
    Args:
        condition: A function that takes an instance of the class and returns True if the fields should be enabled
        fields: A list of fields to be dynamically enabled or disabled
    Example:
        ::
            @some_field.connect
            @enable_if(
                [some_field]
            )
            def _enable_fields(self) -> bool:
                return some_field.value
    """
    # Ideally we could use subclassing to add both the vfield and this event handler, but there
    # seems to be a bug preventing this: https://github.com/hanjinliu/magic-class/issues/113.

    # Start by disabling all the fields

    def _decorator(fn: Callable[[EnabledHandlerType], bool])-> Callable[[EnabledHandlerType], None]:
        for field in fields:
            field.visible = False
            field.enabled = False

        def make_handler(fn: Callable) -> Callable[[EnabledHandlerType], None]:
            def handler(parent: EnabledHandlerType):
                enable = fn(parent)
                if enable:
                    logger.info(f"{parent.__class__} Activated")
                else:
                    logger.info(f"{parent.__class__} Deactivated")
                for field in fields:
                    enable_field(parent, field, enabled=enable)
            return handler

        return make_handler(fn)
    
    return _decorator


class LastDimensionOptions(Enum):
    XYZTC = "XYZTC"
    XYZCT = "XYZCT"
    Metadata = "Get from Metadata"

# class NapariFields(FieldGroup, ABC):
#     def __init__(self, layout: str = "vertical", labels: bool = False, name: str | None = None, **kwargs):
#         super().__init__(layout, labels, name, **kwargs)

# class NapariFieldGroupCompatible(Protocol):
#     from magicclass.widgets import Container
#     from qtpy.QtWidgets import QWidget
#     errors: MagicField[Label]
#     parent: QWidget
#     _widget: Container

class NapariFieldGroup:
    # This implementation is a bit ugly. This is a mixin that can only be used on a `FieldGroup`.
    # However, it can't inherit from FieldGroup because then the metaclass would look for fields in this
    # class definition, find none, and then make an empty GUI page when this is rendered.
    # It also can't inherit from a FieldGroup-like Protocol as mypy suggests for mixin classes
    # (https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes) because it doesn't 
    # implement the attributes of a FieldGroup. Ideally this could be a Protocol subclass as well
    # to make it remain abstract, but the Protocol metaclass interferes with the FieldGroup metaclass
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self = cast(FieldGroup, self)
        # self._widget._mgui_bind_parent_change_callback(self._validate)
        self.changed.connect(self._validate, unique=False)

        # Style the error label. 
        # We have to check this is a QLabel because in theory this might run in a non-QT backend
        errors = self.errors.native
        from qtpy.QtWidgets import QLabel
        if isinstance(errors, QLabel):
            errors.setStyleSheet("color: red;")
            errors.setWordWrap(True)
        # super(Container, self).connect(self._validate)
        # self.connect(self._validate)

    # def _on_value_change(self, *args, **kwargs) -> None:
    #     super()._on_value_change(*args, **kwargs)
    #     self._validate()

    def _get_parent_tab_widget(self: Any) -> QTabWidget:
        return self.parent.parentWidget()

    def _get_tab_index(self: Any) -> int:
        return self._get_parent_tab_widget().indexOf(self._widget._qwidget)

    def _set_valid(self: Any, valid: bool):
        from qtpy.QtGui import QIcon
        tab_parent = self._get_parent_tab_widget()
        index = self._get_tab_index()
            
        if hasattr(self, "fields_enabled") and not self.fields_enabled.value:
            # Special case for "diabled" sections
            tab_parent.setTabIcon(index, QIcon(GREY))
        elif valid:
            tab_parent.setTabIcon(index, QIcon(GREEN))
        else:
            tab_parent.setTabIcon(index, QIcon(RED))

    def _validate(self: Any):
        self.errors.value =  get_friendly_validations(self)
        valid = not bool(self.errors.value)
        self.errors.visible = not valid
        self._set_valid(valid)

    def _make_model(self):
        raise NotImplementedError()

class DeskewFields(NapariFieldGroup, FieldGroup):

    def _get_dimension_options(self, _) -> List[str]:
        """
        Returns the list of dimension order options that might be possible for the current image stack
        """
        default = ["Get from Metadata"]
        try:
            merged = self._merge_layers()
        except Exception:
            return default
        ndims = len(merged._dims_order)
        if ndims == 3:
            return ["ZYX"] + default
        elif ndims == 4:
            return ["TZYX", "CZYX"] + default
        elif ndims == 5:
            return ["TCZYX", "CTZYX"] + default
        else:
            raise Exception("Only 3-5 dimensional arrays are supported")

    img_layer = field(List[Image], widget_type='Select').with_options(label = "Image Layer").with_choices(lambda _x, _y: get_layers(Image))
    pixel_sizes = field(Tuple[float, float, float]).with_options(
        value=(DefinedPixelSizes.get_default("X"), DefinedPixelSizes.get_default("Y"), DefinedPixelSizes.get_default("Z")),
        label="Pixel Sizes (XYZ)"
    )
    angle = field(LatticeData.get_default("angle")).with_options(value=LatticeData.get_default("angle"), label="Skew Angle")
    device = field(str).with_choices(cle.available_device_names()).with_options(label="Graphics Device")
    # merge_all_channels = field(False).with_options(label="Merge all Channels")
    dimension_order = field(str).with_options(value=LastDimensionOptions.Metadata.value).with_choices(_get_dimension_options).with_options(label="Dimension Order")
    skew_dir = field(DeskewDirection.Y).with_options(label = "Skew Direction")
    errors = field(Label).with_options(label="Errors")

    @img_layer.connect
    def _img_changed(self):
        self.dimension_order.reset_choices()

    def _merge_layers(self) -> Image:
        """
        Returns a single image merged from all the selected layers
        """
        from napari.layers.utils.stack_utils import images_to_stack
        if len(self.img_layer.value) == 0:
            raise Exception("At least one image layer must be selected.")
        return images_to_stack(self.img_layer.value)

    def _make_model(self) -> DeskewParams:
        return DeskewParams(
            **lattice_params_from_napari(
                img=self._merge_layers(),
                dimension_order=None if self.dimension_order.value == "Get from Metadata" else self.dimension_order.value,
                physical_pixel_sizes=self.pixel_sizes.value,
            ),
            angle=self.angle.value,
            skew = self.skew_dir.value,
        )

class DeconvolutionFields(NapariFieldGroup, FieldGroup):
    """
    A counterpart to the DeconvolutionParams Pydantic class
    """
    fields_enabled = field(False, label="Enabled")
    decon_processing = field(DeconvolutionChoice, label="Processing Algorithm")
    psf = field(List[Path], label = "PSFs")
    psf_num_iter = field(int, label = "Number of Iterations")
    background = field(ComboBox).with_choices(
        [it.value for it in BackgroundSource]
    ).with_options(label="Background")
    background_custom = field(float).with_options(
        visible=False,
        label="Custom Background"
    )
    errors = field(Label).with_options(label="Errors")


    # @background.connect
    # def _show_custom_background(self):
    #     self.background_custom.visible = self.background == BackgroundSource.Custom
        
    @background.connect
    @enable_if(
        [background_custom]
    )
    def _enable_custom_background(self) -> bool:
        return self.background.value == BackgroundSource.Custom

    # @fields_enabled.connect
    # def _enable_fields(self) -> bool:
    #     self.decon_processing.visible = self.fields_enabled

    @fields_enabled.connect
    @enable_if(
        fields = [
            decon_processing,
            psf,
            psf_num_iter,
            background
        ]
    )
    def _enable_fields(self) -> bool:
        return self.fields_enabled.value

    def _make_model(self) -> Optional[DeconvolutionParams]:
        if not self.fields_enabled.value:
            return None
        if self.background.value == BackgroundSource.Custom:
            background = self.background_custom.value
        elif self.background.value == BackgroundSource.Auto:
            background = "auto"
        else:
            background = "second_last"
        return DeconvolutionParams(
            decon_processing=self.decon_processing.value,
            background=background,
            psf_num_iter=self.psf_num_iter.value
        )

class CroppingFields(NapariFieldGroup, FieldGroup):
    """
    A counterpart to the CropParams Pydantic class
    """
    fields_enabled = field(False, label="Enabled")
    shapes= field(List[Shapes], widget_type="Select", label = "ROIs").with_options(choices=lambda _x, _y: get_layers(Shapes))
    z_range = field(Tuple[int, int]).with_options(
        label = "Z Range",
        value = (0, 1),
        options = dict(
            min = 0,
            max = 1
        ),
    )
    errors = field(Label).with_options(label="Errors")

    @fields_enabled.connect
    @enable_if([shapes, z_range])
    def _enable_workflow(self) -> bool:
        return self.fields_enabled.value


    # roi_layer_list: ShapesData
    # @magicclass(visible=False)
    # class Fields(MagicTemplate):
    #     shapes= vfield(ShapesData, label = "ROIs")#Optional[Shapes] = None
    #     z_range = vfield(Tuple[int, int]).with_options(
    #         label = "Z Range",
    #         value = (0, 1),
    #         options = dict(
    #             min = 0,
    #             max = 1
    #         ),
    #     )
        # _shapes_layer: Optional[Shapes] = None

        # @set_design(font_size=10, text="Click to activate Cropping Layer", background_color="magenta")
        # @click(enables=["Import_ImageJ_ROI", "Crop_Preview"])
        # @set_design(text="New Cropping Layer")
        # def activate_cropping(self):
        #     self._shapes_layer = self.parent_viewer.add_shapes(shape_type='polygon', edge_width=1, edge_color='white',
        #                                                         face_color=[1, 1, 1, 0], name="Cropping BBOX layer")
        #     # TO select ROIs if needed
        #     self._shapes_layer.mode = "SELECT"
        #     self["activate_cropping"].text = "Cropping layer active"
        #     self["activate_cropping"].background_color = "green"

    def _make_model(self) -> Optional[CropParams]:
        import numpy as np
        if self.fields_enabled.value:
            return CropParams(
                z_start=self.z_range.value[0],
                z_end=self.z_range.value[1],
                roi_layer_list=ShapesData([np.array(shape.data) for shape in self.shapes.value])
            )
        return None

class WorkflowFields(NapariFieldGroup, FieldGroup):
    """
    Handles the workflow related parameters
    """
    fields_enabled = field(False, label="Enabled")
    workflow_source = field(ComboBox).with_options(label = "Workflow Source").with_choices([it.value for it in WorkflowSource])
    workflow_path = field(Path).with_options(label = "Workflow Path", visible=False)
    errors = field(Label).with_options(label="Errors")

    @fields_enabled.connect
    @enable_if([workflow_source])
    def _enable_workflow(self) -> bool:
        return self.fields_enabled.value

    @fields_enabled.connect
    @enable_if([workflow_path])
    def _workflow_path(self) -> bool:
        return self.workflow_source.value == WorkflowSource.CustomPath

    def _make_model(self) -> Optional[Workflow]:
        from lls_core.workflow import import_workflow_modules
        from napari_workflows._io_yaml_v1 import load_workflow
        if not self.fields_enabled.value:
            return None
        if self.workflow_source.value == WorkflowSource.ActiveWorkflow:
            return WorkflowManager.install(self.parent_viewer).workflow
        else:
            import_workflow_modules(self.workflow_path.value)
            return load_workflow(str(self.workflow_path.value))

# @magicclass(name="5. Output")
class OutputFields(NapariFieldGroup, FieldGroup):
    set_logging = field(Log_Levels.INFO).with_options(label="Logging Level")
    time_range = field(Tuple[int, int]).with_options(
        label="Time Export Range",
        value=(0, 1),
        options = dict(
            min=0,
            max=100,
        )
    )
    channel_range = field(Tuple[int, int]).with_options(
        label="Channel Range",
        value=(0, 1),
        options = dict(
            min=0,
            max=100,
        )
    )
    save_type = field(SaveFileType).with_options(
        label = "Save Format"
    )
    save_path = field(Path).with_options(
        label = "Save Directory",
        value = Path(history.get_save_history()[0])
    )
    save_name = field(str).with_options(
        label = "Save Prefix"
    )
    errors = field(Label).with_options(label="Errors")

    def _make_model(self) -> OutputParams:
        return OutputParams(
            channel_range=range(self.channel_range.value[0], self.channel_range.value[1]),
            time_range=range(self.time_range.value[0], self.time_range.value[1]),
            save_dir=self.save_path.value,
            save_name=self.save_name.value,
            save_type=self.save_type.value
        )


# @DeskewWidget.img_layer.connect
