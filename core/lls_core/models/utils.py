
from typing import Type
from enum import Enum
from pydantic import BaseModel


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
    def get_default(cls, field_name: str):
        return cls.__fields__[field_name].get_default()

    @classmethod
    def get_description(cls, field_name: str) -> str:
        return cls.__fields__[field_name].field_info.description
