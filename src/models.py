"""Input schemas and Qwen chat-token definitions used by the decoder."""

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


t_token_ids = list[int]


class ChatTokenIds(BaseModel):
    """Validate a message whose role and content are already tokenized.

    Attributes:
        role: Token IDs identifying the message's role.
        content: Token IDs containing the message body.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    role: t_token_ids
    content: t_token_ids


class FnType(BaseModel):
    """Validate a supported parameter or return type.

    Attributes:
        type: Supported scalar: string, number, integer, or boolean.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "integer", "boolean"]


class Function(BaseModel):
    """Describe a function that the language model may select.

    Attributes:
        name: Function name to include in the output.
        description: Explanation of the function's purpose for the model.
        parameters: Required parameter names mapped to their types.
        returns: Type of the function's return value.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, FnType]
    returns: FnType

    def dump_json(self) -> str:
        """Serialize the definition for inclusion in the system prompt.

        Returns:
            The function definition as compact JSON text.
        """
        return self.model_dump_json()


class Prompt(BaseModel):
    """Validate one natural-language request.

    Attributes:
        prompt: Request to translate into a function call.
    """

    model_config = ConfigDict(extra="forbid")
    prompt: str


# Qwen3 token constants do not need classes or validation.
ROLE_IDS: dict[str, t_token_ids] = {
    "system": [8948],
    "user": [872],
    "assistant": [77091],
}

CONTROL_IDS: dict[str, t_token_ids] = {
    "start": [151644],
    "end": [151645],
    "think": [151667],
    "end_think": [151668],
    "newline": [198],
    "double_newline": [271],
}
