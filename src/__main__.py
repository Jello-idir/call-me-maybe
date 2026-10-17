from llm_sdk import Small_LLM_Model
from transformers import AutoTokenizer
from pydantic import BaseModel, ConfigDict
from typing import Literal
import json
import sys
import numpy as np
import time
from enum import Enum
from typing import TypedDict
import argparse

USERCOLOR = "\033[1;34m"
ASSISTANTCOLOR = "\033[1;33m"
RESETCOLOR = "\033[0m"

# 151643 = <|endoftext|>
# 151644 = <|im_start|>
# 151645 = <|im_end|>
# 151646 = <|object_ref_start|>
# 151647 = <|object_ref_end|>
# 151648 = <|box_start|>
# 151649 = <|box_end|>
# 151650 = <|quad_start|>
# 151651 = <|quad_end|>
# 151652 = <|vision_start|>
# 151653 = <|vision_end|>
# 151654 = <|vision_pad|>
# 151655 = <|image_pad|>
# 151656 = <|video_pad|>
# 151657 = <tool_call>
# 151658 = </tool_call>
# 151659 = <|fim_prefix|>
# 151660 = <|fim_middle|>
# 151661 = <|fim_suffix|>
# 151662 = <|fim_pad|>
# 151663 = <|repo_name|>
# 151664 = <|file_sep|>
# 151665 = <tool_response>
# 151666 = </tool_response>
# 151667 = <think>
# 151668 = </think>


t_token_ids = list[int]

class Chat(TypedDict):
    role: str
    content: str

class ChatTokenIds(TypedDict):
    role: t_token_ids
    content: t_token_ids


class FnType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "boolean"]


class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, FnType]
    returns: FnType

    def print_json(self):
        print(json.dumps(self.model_dump(), indent=4))


class Prompt(BaseModel):
    prompt: str


class Roles(Enum):
    SYSTEM = ("system", 8948)
    USER = ("user", 872)
    ASSISTANT = ("assistant", 77091)

    def __str__(self) -> str:
        return self.value[0]

    def ids(self) -> t_token_ids:
        return [self.value[1]]


class ControlToken(Enum):
    EOD = ("<|endoftext|>", 151643)
    START = ("<|im_start|>", 151644)
    END = ("<|im_end|>", 151645)
    THINK = ("<think>", 151667)
    END_THINK = ("</think>", 151668)
    NL = ("\n", 198)
    DOUBLE_NL = ("\n\n", 271)

    def __str__(self) -> str:
        return self.value[0]

    def ids(self) -> t_token_ids:
        return [self.value[1]]


def log_io_in_file(title: str, line: str, filename: str = "log.txt") -> None:
    with open(filename, "a") as file:
        file.write(f"\033[1;33m{title}\033[0m\n")
        file.write(f"\033[33m--------------------------\033[0m\n")
        file.write(f"{line}\n")
        file.write(f"\033[33m--------------------------\033[0m\n")
        if title == "output:":
            file.write(f"\n\n")


def format_message_str(chats: list[Chat]) -> str:

    result = ""
    for chat in chats:
        if not isinstance(chat, dict) or len(chat) != 2:
            raise ValueError("Each message must be a dict with 'role' and 'content' keys.")
        if "role" not in chat or "content" not in chat:
            raise ValueError("Each message must have 'role' and 'content' keys.")
        result += (
            f"{ControlToken.START}{chat['role']}\n"
            f"{chat['content']}{ControlToken.END}\n"
        )

    return result


def format_token_ids_mssg(chats: list[ChatTokenIds]) -> t_token_ids:

    result: list[int] = []
    for chat in chats:
        if not isinstance(chat, dict) or len(chat) != 2:
            raise ValueError("Each message must be a dict with 'role' and 'content' keys.")
        if "role" not in chat or "content" not in chat:
            raise ValueError("Each message must have 'role' and 'content' keys.")
        if not isinstance(chat["role"], list) or not isinstance(chat["content"], list):
            raise ValueError("Both 'role' and 'content' must be lists of token IDs.")
        if not all(isinstance(i, int) for i in chat["role"]):
            raise ValueError("'role' must be a list of integers.")
        if not all(isinstance(i, int) for i in chat["content"]):
            raise ValueError("'content' must be a list of integers.")
        result.extend(ControlToken.START.ids())
        result.extend(chat["role"])
        result.extend(ControlToken.NL.ids())
        result.extend(chat["content"])
        result.extend(ControlToken.END.ids())
        result.extend(ControlToken.NL.ids())
    return result


def prefix_assistant_token_ids(thinking: bool = False) -> t_token_ids:
    if thinking:
        return (
            ControlToken.START.ids()
            + Roles.ASSISTANT.ids()
            + ControlToken.NL.ids()
            + ControlToken.THINK.ids()
            + ControlToken.NL.ids()
        )
    return (
        ControlToken.START.ids()
        + Roles.ASSISTANT.ids()
        + ControlToken.NL.ids()
        + ControlToken.THINK.ids()
        + ControlToken.DOUBLE_NL.ids()
        + ControlToken.END_THINK.ids()
        + ControlToken.DOUBLE_NL.ids()
    )


def next_id(llm: Small_LLM_Model, input_ids: list[int]) -> int:
    logits = llm.get_logits_from_input_ids(input_ids)
    index_of_max_logit = np.argmax(logits)
    return int(index_of_max_logit)


def generate_response(
        llm: Small_LLM_Model,
        input_ids: list[int],
        max_tokens: int = 256,
        print_stdout: bool = True
        ) -> list[int]:

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    stop_ids = ControlToken.END.ids() + ControlToken.EOD.ids()

    just_started = True
    generated_ids: t_token_ids = []
    for _ in range(max_tokens):
        new_id = next_id(llm, input_ids + generated_ids)

        if print_stdout:
            if just_started and new_id in ControlToken.DOUBLE_NL.ids():
                just_started = False
            else:
                print(llm.decode([new_id]), end="", flush=True)

        if just_started:
            just_started = False

        generated_ids.append(new_id)
        if new_id in stop_ids:
            break

    if new_id not in stop_ids:
        print("Warning: Reached max_tokens limit without encountering a stop token.\n", file=sys.stderr)
        generated_ids.extend(ControlToken.END.ids())

    return generated_ids


def generate_thoughts(
        llm: Small_LLM_Model,
        input_ids: list[int],
        max_tokens: int = 256,
        print_thoughts: bool = True
    ) -> list[int]:

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    stop_ids = (
        ControlToken.END_THINK.ids()
        + ControlToken.END.ids()
        + ControlToken.EOD.ids()
    )

    if print_thoughts:
        print("\033[2m", end="", flush=True)

    generated_ids: t_token_ids = []
    for _ in range(max_tokens):
        new_id = next_id(llm, input_ids + generated_ids)

        generated_ids.append(new_id)
        if new_id in stop_ids:
            break

        if print_thoughts:
            if new_id not in ControlToken.DOUBLE_NL.ids():
                print(llm.decode([new_id]), end="", flush=True)

    if print_thoughts:
        print("\033[0m", end="", flush=True)

    if new_id not in stop_ids:
        generated_ids.extend(ControlToken.END_THINK.ids())

    return generated_ids


def start_chat(llm: Small_LLM_Model,
               brainwash: str,
               max_tokens: int = 512,
               print_thinking: bool = False) -> None:
    serialized_prompt: t_token_ids = []

    assistant_prefix: t_token_ids = prefix_assistant_token_ids(thinking=print_thinking)

    # adding brainwash as system message
    serialized_prompt.extend(format_token_ids_mssg([{
            "role": Roles.SYSTEM.ids(),
            "content": llm.encode(brainwash)[0].tolist()}]))

    while True:
        user_mssg = input(f"\n{USERCOLOR}you: {RESETCOLOR}")

        if user_mssg == "/exit":
            break

        # adding user message to the serialized prompt
        user_token_ids = format_token_ids_mssg(
            [
                {
                    "role": Roles.USER.ids(),
                    "content": llm.encode(user_mssg)[0].tolist()
                }
            ]
        )

        serialized_prompt.extend(user_token_ids)
        serialized_prompt.extend(assistant_prefix)

        # generating thoughts
        thoughts_ids: t_token_ids = []
        if print_thinking:
            thoughts_ids = generate_thoughts(
                llm, serialized_prompt,
                max_tokens=max_tokens,
                print_thoughts=print_thinking
            )

        print(f"{ASSISTANTCOLOR}assistant: {RESETCOLOR}", end="", flush=True)

        # generating answer
        generated_answer_ids: t_token_ids = generate_response(
            llm, serialized_prompt + thoughts_ids,
            max_tokens=max_tokens,
            print_stdout=True
        )
        generated_answer_ids.extend(ControlToken.NL.ids())
        serialized_prompt.extend(generated_answer_ids)


def main() -> None:

    # mandatory args
    #   --functions_definition
    #   --input
    #   --output

    # optional args
    # --debug (not implimented yet)
    # --thinking



    with open("brainwash.md") as bw_file:
        brainwash = bw_file.read()

    llm = Small_LLM_Model()
    # damn = ["system", "user", "assitant"]

    # for d in damn:
    #     print(d)
    #     print(llm.encode(d))
    #     print()

    print("-----------------------\n\n")
    start_chat(llm, brainwash, max_tokens=512, print_thinking=True)



if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        sys.stderr.write(f"\033[1;31mError: \033[0m{err}\n"  )
    except KeyboardInterrupt:
        sys.stderr.write("\n\033[1;31mKeyboardInterrupt: \033[0mExiting...\n")
