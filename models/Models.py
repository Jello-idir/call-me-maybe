"""Input schemas and Qwen chat-token definitions."""

from pydantic import BaseModel, ConfigDict
from typing import TypedDict, Literal
from enum import Enum
import json


t_token_ids = list[int]


class ChatTokenIds(TypedDict):
    """Describe a message whose role and content are already tokenized.

    Attributes:
        role: Token IDs identifying the message's role.
        content: Token IDs containing the message body.
    """

    role: t_token_ids
    content: t_token_ids


class FnType(BaseModel):
    """Validate a supported parameter or return type.

    Attributes:
        type: JSON scalar type: string, number, or boolean.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "boolean"]


class Function(BaseModel):
    """Describe a function that the language model may select.

    Attributes:
        name: Function name to include in the output.
        description: Explanation of the function's purpose for the model.
        parameters: Required parameter names mapped to their types.
        returns: Type of the function's return value.
    """

    name: str
    description: str
    parameters: dict[str, FnType]
    returns: FnType

    def dump_json(self) -> str:
        """Serialize the definition for inclusion in the system prompt.

        Returns:
            The function definition as indented JSON text.
        """
        return json.dumps(self.model_dump(), indent=4)


class Prompt(BaseModel):
    """Validate one natural-language request.

    Attributes:
        prompt: Request to translate into a function call.
    """

    model_config = ConfigDict(extra="forbid")
    prompt: str


class Roles(Enum):
    """Store Qwen token IDs for system, user, and assistant roles."""

    SYSTEM = 8948
    USER = 872
    ASSISTANT = 77091

    def ids(self) -> t_token_ids:
        """Wrap the role token ID in a list.

        Returns:
            A one-element list containing this role's token ID.
        """
        return [self.value]


class ControlToken(Enum):
    """Store Qwen message, thinking, and newline token IDs."""

    START = 151644
    END = 151645
    THINK = 151667
    END_THINK = 151668
    NL = 198
    DOUBLE_NL = 271

    def ids(self) -> t_token_ids:
        """Wrap the control token ID in a list.

        Returns:
            A one-element list containing this control token's ID.
        """
        return [self.value]
