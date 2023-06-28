from typing import Any, Callable, overload
from magicclass.stylesheets import StyleSheet
from magicclass._gui import MenuGuiBase
from magicclass._gui.mgui_ext import Clickable
from magicclass._gui._function_gui import FunctionGuiPlus
from magicclass.types import WidgetTypeStr, PopUpModeStr, ErrorModeStr
from magicclass.help import HelpWidget
from macrokit import Macro
from magicclass._gui._base import (
    PopUpMode,
    ErrorMode,
    defaults,
    MagicTemplate,
    check_override,
    convert_attributes,
)
from magicclass.types import WidgetType
from magicclass._gui.class_gui import (
    ClassGuiBase,
    ClassGui,
)


# If the user does pass a class into the decorator
@overload
def magicclass(
    class_: type,
    *,
    layout: str | None = ...,
    labels: bool = ...,
    name: str | None = ...,
    visible: bool | None = ...,
    close_on_run: bool | None = ...,
    popup_mode: PopUpModeStr | PopUpMode = ...,
    error_mode: ErrorModeStr | ErrorMode = ...,
    widget_type: WidgetTypeStr | WidgetType = ...,
    icon: Any = ...,
    stylesheet: str | StyleSheet | None = ...,
    properties: dict[str, Any] | None = ...,
    record: bool = ...,
    symbol: str = ...,
) -> type[ClassGui]:
    ...

# If the user doesn't pass a class into the decorator
@overload
def magicclass(
    *,
    layout: str | None = ...,
    labels: bool = ...,
    name: str | None = ...,
    visible: bool | None = ...,
    close_on_run: bool | None = ...,
    popup_mode: PopUpModeStr | PopUpMode = ...,
    error_mode: ErrorModeStr | ErrorMode = ...,
    widget_type: WidgetTypeStr | WidgetType = ...,
    icon: Any = ...,
    stylesheet: str | StyleSheet | None = ...,
    properties: dict[str, Any] | None = ...,
    record: bool = ...,
    symbol: str = ...,
) -> Callable[[type], type[ClassGui]]:
    ...