
from typing import Any, Type
from enum import Enum
from pydantic import BaseModel
from typer import Option
from typer.models import OptionInfo


def enum_choices(enum: Type[Enum]) -> str:
    """
    Returns a human readable list of enum choices
    """
    return "{" +  ", ".join([it.name for it in enum]) + "}"

class FieldAccessMixin(BaseModel):
    """
    Adds methods to a BaseModel for accessing useful field information
    """

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
