from __future__ import annotations

from dataclasses import dataclass
from typing import  Callable, TYPE_CHECKING
from operator import attrgetter

if TYPE_CHECKING:
    from magicclass import MagicTemplate
    from magicgui.widgets.bases import ValueWidget


@dataclass
class ParentConnect:
    """
    A function that wants to be connected to a parent or sibling's field.
    This will be resolved after the GUI is instantiated.
    """
    path: str
    func: Callable

    def resolve(self, root: MagicTemplate, event_owner: MagicTemplate, field_name: str) -> None:# -> Callable[..., Any]:
        """
        Converts this object into a true function that is connected to the appropriate change event
        """
        field: ValueWidget = attrgetter(self.path)(root)
        # field_owner = field.parent
        bound_func = field.changed.connect(lambda: self.func(event_owner, field.value))
        setattr(event_owner, field_name, bound_func)

def connect_parent(path: str) -> Callable[..., ParentConnect]:
    """
    Mark this function as wanting to connect to a parent or sibling event
    """
    def decorator(fn: Callable) -> ParentConnect:
        return ParentConnect(
            path=path,
            func=fn
        )
    return decorator
