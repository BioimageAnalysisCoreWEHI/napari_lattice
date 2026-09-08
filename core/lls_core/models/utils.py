
from typing import Any, Tuple, Type, TypeVar, Union
from typing_extensions import Self
from enum import Enum
from pydantic.v1 import BaseModel, Extra
from contextlib import contextmanager

T = TypeVar("T")
def as_tuple(x: Union[Tuple[T], T]) -> Tuple[T]:
    """
    Converts the results to a tuple if they weren't already
    """
    return x if isinstance(x, tuple) else (x,)

def enum_choices(enum: Type[Enum]) -> str:
    """
    Returns a human readable list of enum choices
    """
    return "{" +  ", ".join([f'"{it.name}"' for it in enum]) + "}"

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

class FieldAccessModel(BaseModel):
    """
    Adds methods to a BaseModel for accessing useful field information
    """
    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True
        validate_assignment = True

    @classmethod
    def get_default(cls, field_name: str) -> Any:
        """
        Shortcut method for returning the default value of a given field
        """
        return cls.__fields__[field_name].get_default()

    @classmethod
    def get_description(cls, field_name: str) -> str:
        """
        Shortcut method for returning the description of a given field
        """
        return cls.__fields__[field_name].field_info.description

    @classmethod
    def to_definition_dict(cls) -> dict:
        """
        Recursively converts the model into a dictionary whose keys are field names and whose
        values are field descriptions. This is used to document the model to users
        """
        ret = {}
        for key, value in cls.__fields__.items():
            if value.field_info.extra.get("cli_hide"):
                # Hide any fields that have cli_hide = True
                continue

            if isinstance(value.outer_type_, type) and issubclass(value.outer_type_, FieldAccessModel):
                rhs = value.outer_type_.to_definition_dict()
            else:
                rhs: str
                if "cli_description" in value.field_info.extra:
                    # cli_description can be used to configure the help text that appears for fields for the CLI only
                    rhs = value.field_info.extra["cli_description"]
                else:
                    rhs = value.field_info.description
                rhs += f" Default: {value.get_default()}."
            ret[key] = rhs
        return ret

    def copy_validate(self, **kwargs) -> Self:
        """
        Like `.copy()`, but validates the results.
        See https://github.com/pydantic/pydantic/issues/418 for more information
        """
        updated = self.copy(**kwargs)
        return updated.validate(updated.dict())

    @classmethod
    def make(cls, validate: bool = True, **kwargs: Any):
        """
        Creates an instance of this class, with validation either enabled or disabled 
        """
        if validate:
            return cls(**kwargs)
        else:
            return cls.construct(**kwargs)
