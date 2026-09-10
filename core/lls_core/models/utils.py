
from typing import Any, Tuple, Type, TypeVar, Union, get_args, get_origin
from typing_extensions import Self
from enum import Enum
from contextlib import contextmanager
from pydantic import BaseModel, ConfigDict

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
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, validate_assignment=True)

    @classmethod
    def get_default(cls, field_name: str) -> Any:
        """
        Shortcut method for returning the default value of a given field
        """
        return cls.model_fields[field_name].get_default()

    @classmethod
    def get_description(cls, field_name: str) -> str:
        """
        Shortcut method for returning the description of a given field
        """
        return cls.model_fields[field_name].description

    @classmethod
    def to_definition_dict(cls) -> dict:
        """
        Recursively converts the model into a dictionary whose keys are field names and whose
        values are field descriptions. This is used to document the model to users
        """
        ret = {}
        for key, value in cls.model_fields.items():
            extra = value.json_schema_extra or {}
            if extra.get("cli_hide"):
                # Hide any fields that have cli_hide = True
                continue

            annotation = value.annotation
            if get_origin(annotation) is Union:
                non_none = [a for a in get_args(annotation) if a is not type(None)]
                if len(non_none) == 1:
                    annotation = non_none[0]

            if isinstance(annotation, type) and issubclass(annotation, FieldAccessModel):
                rhs = annotation.to_definition_dict()
            else:
                rhs: str
                if "cli_description" in extra:
                    # cli_description can be used to configure the help text that appears for fields for the CLI only
                    rhs = extra["cli_description"]
                else:
                    rhs = value.description
                rhs += f" Default: {value.get_default()}."
            ret[key] = rhs
        return ret

    def copy_validate(self, **kwargs) -> Self:
        """
        Like `.copy()`, but validates the results.
        See https://github.com/pydantic/pydantic/issues/418 for more information
        """
        updated = self.model_copy(**kwargs)
        return type(self).model_validate(updated.model_dump())

    @classmethod
    def make(cls, validate: bool = True, **kwargs: Any):
        """
        Creates an instance of this class, with validation either enabled or disabled 
        """
        if validate:
            return cls(**kwargs)
        else:
            return cls.model_construct(**kwargs)
