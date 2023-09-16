
from typing import Any, Type
from enum import Enum
from pydantic import BaseModel, Extra
from typer import Option
from contextlib import contextmanager

def is_pathlike(x: Any) -> bool:
    from os import PathLike
    return isinstance(x, (str, bytes, PathLike))

def enum_choices(enum: Type[Enum]) -> str:
    """
    Returns a human readable list of enum choices
    """
    return "{" +  ", ".join([it.name for it in enum]) + "}"

@contextmanager
def ignore_keyerror():
    """
    Context manager that ignores KeyErrors from missing fields.
    This allows for the validation to continue even if a single field
    is missing, eventually resulting in a more user-friendly error message
    """
    try:
        yield
    except KeyError:
        pass

class FieldAccessMixin(BaseModel):
    """
    Adds methods to a BaseModel for accessing useful field information
    """
    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @classmethod
    def get_default(cls, field_name: str) -> Any:
        return cls.__fields__[field_name].get_default()

    @classmethod
    def get_description(cls, field_name: str) -> str:
        return cls.__fields__[field_name].field_info.description

    @classmethod
    def make_typer_field(cls, field_name: str, extra_description: str = "") -> Any:
        field = cls.__fields__[field_name]
        return Option(
            default = field.get_default(),
            help=field.field_info.description + extra_description
        )
