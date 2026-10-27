"""Generate JSON function calls from prompts using constrained decoding."""

from llm_sdk import Small_LLM_Model
from src.models import (
    ChatTokenIds, CONTROL_IDS, FnType, Function, Prompt, ROLE_IDS, t_token_ids,
)
import json
import sys
import numpy as np
import argparse
from pathlib import Path
from src import visualization


def write_error(notice: str, message: str) -> None:
    """Print a labelled error to standard error.

    Args:
        notice: Short label displayed before the error.
        message: Explanation of the error.
    """
    sys.stderr.write(f"\033[1;31m{notice}: \033[0m{message}\n")
    sys.stderr.flush()


def load_file(filename: str) -> str:
    """Read the complete contents of a text file.

    Args:
        filename: Path to the file.

    Returns:
        The file contents as a string.

    Raises:
        ValueError: The file is missing or cannot be read.
    """

    try:
        with open(filename, encoding="utf-8") as f:
            data = f.read()
    except FileNotFoundError:
        raise ValueError(f"File not found: {filename}")
    except Exception as e:
        raise ValueError(f"Error reading file {filename}: {e}")
    return data


def format_token_ids_mssg(chats: list[dict[str, t_token_ids]]) -> t_token_ids:
    """Wrap tokenized messages in Qwen's chat markers.

    Args:
        chats: Messages containing role and content token IDs.

    Returns:
        One token sequence containing all messages and their markers.

    Raises:
        pydantic.ValidationError: A message has invalid keys or token IDs.
    """

    result: list[int] = []

    for chat in chats:
        message = ChatTokenIds.model_validate(chat)
        result.extend(CONTROL_IDS["start"])
        result.extend(message.role)
        result.extend(CONTROL_IDS["newline"])
        result.extend(message.content)
        result.extend(CONTROL_IDS["end"])
        result.extend(CONTROL_IDS["newline"])

    return result


def prefix_assistant_token_ids() -> t_token_ids:
    """Build the assistant prefix with an empty thinking section.

    Returns:
        Token IDs that start the assistant's final answer.
    """
    return (
        CONTROL_IDS["start"]
        + ROLE_IDS["assistant"]
        + CONTROL_IDS["newline"]
        + CONTROL_IDS["think"]
        + CONTROL_IDS["double_newline"]
        + CONTROL_IDS["end_think"]
        + CONTROL_IDS["double_newline"]
    )


def init_arg_parser() -> argparse.Namespace:
    """Parse file paths and the optional visualization flag.

    Returns:
        Parsed command-line options, including their default paths.

    Raises:
        SystemExit: Help was requested or an argument is invalid.
    """

    arg_parser = argparse.ArgumentParser(
        description="Convert prompts into JSON function calls."
    )

    arg_parser.add_argument(
        "--functions_definition",
        type=str,
        required=False,
        help="Path to the JSON file containing function definitions.",
        default="data/input/functions_definition.json"
    )

    arg_parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to the input file.",
        default="data/input/function_calling_tests.json"
    )

    arg_parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to the output file.",
        default="data/output/function_calls.json",
    )

    arg_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show generation steps in the terminal (on standard error).",
    )

    return arg_parser.parse_args()


def load_prompts(filename: str) -> list[Prompt]:
    """Load and validate a JSON array of prompts.

    Args:
        filename: Path to the input JSON file.

    Returns:
        Validated prompt objects in input order.

    Raises:
        OSError: The file cannot be opened or read.
        json.JSONDecodeError: The file contains invalid JSON.
        ValueError: The JSON root is not a list.
        pydantic.ValidationError: A prompt does not match its schema.
    """
    with open(filename, encoding="utf-8") as f:
        j = json.load(f)

    if not isinstance(j, list):
        raise ValueError("Input JSON must be a list of prompts.")

    prompts: list[Prompt] = []
    for prompt in j:
        prompts.append(Prompt.model_validate(prompt))

    return prompts


def load_functions(filename: str) -> list[Function]:
    """Load and validate the available function definitions.

    Args:
        filename: Path to the function-definition JSON file.

    Returns:
        Validated function objects in input order.

    Raises:
        OSError: The file cannot be opened or read.
        json.JSONDecodeError: The file contains invalid JSON.
        ValueError: The JSON root is not a non-empty list, or names repeat.
        pydantic.ValidationError: A function does not match its schema.
    """
    with open(filename, encoding="utf-8") as f:
        j = json.load(f)

    if not isinstance(j, list) or not j:
        raise ValueError("Input JSON must be a non-empty list of functions.")

    functions: list[Function] = []
    names: set[str] = set()
    for func in j:
        function = Function.model_validate(func)
        if function.name in names:
            raise ValueError(f"Duplicate function name: {function.name}")
        names.add(function.name)
        functions.append(function)

    return functions


def generate_allowed_tokens(
        llm: Small_LLM_Model,
        serialized_prompt: t_token_ids,
        allowed_sequences: list[t_token_ids]) -> t_token_ids:
    """Select one allowed sequence using the model's next-token scores.

    Args:
        llm: SDK model supplying token scores.
        serialized_prompt: Context tokens preceding the selection.
        allowed_sequences: Non-empty token sequences the model may choose.

    Returns:
        The selected complete token sequence.

    Raises:
        ValueError: Choices are empty, no choice matches the prefix, or no
            allowed next token has a finite score.
    """

    if not allowed_sequences or any(not ids for ids in allowed_sequences):
        raise ValueError("Expected non-empty allowed token sequences")

    generated_ids: t_token_ids = []

    while True:
        position = len(generated_ids)
        allowed_next_ids = sorted({
            ids[position]
            for ids in allowed_sequences
            if len(ids) > position and ids[:position] == generated_ids
        })
        if not allowed_next_ids:
            raise ValueError("No allowed sequence matches the token prefix")

        # A forced token needs no prediction; ask the model only for choices.
        if len(allowed_next_ids) == 1:
            next_token = allowed_next_ids[0]
        else:
            logits = llm.get_logits_from_input_ids(
                serialized_prompt + generated_ids
            )
            allowed_next_ids = [
                token for token in allowed_next_ids
                if np.isfinite(logits[token])
            ]
            if not allowed_next_ids:
                raise ValueError("No valid next token has a finite score")
            next_token = max(allowed_next_ids, key=lambda token: logits[token])

        generated_ids.append(next_token)
        if visualization.enabled:
            visualization.show("Name", llm.decode(generated_ids))
        if generated_ids in allowed_sequences:
            return generated_ids


def generate_function_name(
        llm: Small_LLM_Model,
        serialized_prompt: t_token_ids,
        allowed_functions: dict[str, t_token_ids]) -> t_token_ids:
    """Choose a function name while distinguishing overlapping names.

    Args:
        llm: SDK model supplying token scores and encoding.
        serialized_prompt: Context ending before the function-name content.
        allowed_functions: Function names mapped to their encoded content.

    Returns:
        The chosen name's token IDs, without the closing quote and comma.

    Raises:
        ValueError: No allowed function name can be generated.
    """

    name_ending = llm.encode('",')[0].tolist()
    generated_name = generate_allowed_tokens(
        llm,
        serialized_prompt,
        [ids + name_ending for ids in allowed_functions.values()],
    )
    return generated_name[:-len(name_ending)]


def find_delimiter(text: str, delimiter: str) -> int | None:
    """Find a separator outside quotes, respecting escaped quotes.

    Args:
        text: Decoded value text to inspect.
        delimiter: Separator character to find, usually a comma or brace.

    Returns:
        The separator's index, or None if it has not appeared outside quotes.
    """
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == '\\' and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif char == delimiter and not in_string:
            return index
    return None


def valid_value_prefix(
        text: str, value_type: str, complete: bool = False) -> bool:
    """Check a value, allowing unfinished text while tokens are arriving.

    Temporary endings let json.loads check unfinished numbers and strings.
    These endings are used only for validation, never for generated output.

    Args:
        text: Candidate value text without its separator.
        value_type: Expected JSON type: string, number, or boolean.
        complete: Whether the value must already be complete.

    Returns:
        Whether the text passes the expected type and JSON syntax checks.
    """
    expected_types = {
        "string": (str,), "number": (int, float), "boolean": (bool,),
    }
    ending_possibilities = [""]
    if not complete:
        if not text.strip(" \t\r\n"):
            return True
        if value_type == "boolean":
            if any(word.startswith(text.lstrip(" \t\r\n"))
                   for word in ("true", "false")):
                return True
        elif value_type == "number":
            ending_possibilities += ["0"]
        elif value_type == "string":
            ending_possibilities += ['"', '\\"', '0"', '00"', '000"', '0000"']

    for ending in ending_possibilities:
        try:
            value = json.loads(text + ending)
            if type(value) not in expected_types[value_type]:
                continue
            if value_type == "number" and not np.isfinite(float(value)):
                continue
            if value_type == "string" and not ending:
                value.encode("utf-8")
            return True
        except (ValueError, OverflowError):
            continue
    return False


def generate_value(
        llm: Small_LLM_Model,
        serialized_prompt: t_token_ids,
        value_type: str = "string",
        delimiter: str = "}") -> t_token_ids:
    """Choose only tokens that keep a valid value, stopping at its separator.

    Candidates are checked before they enter the generated context. There
    is no fixed token limit; a valid separator ends generation.

    Args:
        llm: SDK model supplying encoding, decoding, and token scores.
        serialized_prompt: Context ending immediately before the value.
        value_type: Expected JSON type: string, number, or boolean.
        delimiter: Comma for another parameter, or brace for the final one.

    Returns:
        Token IDs for the validated value and its separator.

    Raises:
        ValueError: The type or separator is unsupported, or no valid next
            token has a finite score.
    """
    if value_type not in ("string", "number", "boolean"):
        raise ValueError("Unsupported value type")
    if delimiter not in (",", "}"):
        raise ValueError("Unsupported value type or delimiter")
    generated_value: t_token_ids = []

    while True:
        logits = llm.get_logits_from_input_ids(
            serialized_prompt + generated_value
        )
        for token_id in np.argsort(logits)[::-1]:
            token_id = int(token_id)
            if not np.isfinite(logits[token_id]) or not llm.decode([token_id]):
                continue
            text = llm.decode(generated_value + [token_id])
            end = find_delimiter(text, delimiter)
            if end is not None and text[end + 1:].strip(" \t\r\n"):
                continue
            value_text = text if end is None else text[:end]
            if not valid_value_prefix(value_text, value_type, end is not None):
                continue

            generated_value.append(token_id)
            visualization.show("Value", text)
            if end is not None:
                value_ids: t_token_ids = llm.encode(text[:end + 1])[0].tolist()
                return value_ids
            break
        else:
            raise ValueError(f"No valid next token for JSON {value_type}")


def generate_parameters(
        llm: Small_LLM_Model,
        serialized_prompt: t_token_ids,
        allowed_parameters: dict[str, FnType]) -> t_token_ids:
    """Generate every required parameter exactly once.

    Args:
        llm: SDK model used to choose names and values.
        serialized_prompt: Context immediately before the parameter object.
        allowed_parameters: Required parameter names and their JSON types.

    Returns:
        Token IDs for the complete parameter object, including braces.

    Raises:
        ValueError: A parameter name or value cannot be generated.
    """

    parameter_ids = {
        name: llm.encode(json.dumps(name) + ":")[0].tolist()
        for name in allowed_parameters
    }
    needed_parameters = list(parameter_ids.values())

    # starting the JSON object
    generated_parameters: t_token_ids = llm.encode("{")[0].tolist()
    if not needed_parameters:
        return generated_parameters + llm.encode("}")[0].tolist()

    for _ in range(len(needed_parameters)):
        generated_param = generate_allowed_tokens(
            llm,
            serialized_prompt + generated_parameters,
            needed_parameters,
        )
        needed_parameters.remove(generated_param)
        generated_parameters.extend(generated_param)
        parameter_name = next(
            name for name, ids in parameter_ids.items()
            if ids == generated_param
        )
        visualization.show(
            "Parameter",
            f"{parameter_name} ({allowed_parameters[parameter_name].type})",
        )
        generated_parameters.extend(generate_value(
            llm,
            serialized_prompt + generated_parameters,
            value_type=allowed_parameters[parameter_name].type,
            delimiter="," if needed_parameters else "}",
        ))

    return generated_parameters


def constrained_decoding(
        llm: Small_LLM_Model,
        brainwash: str,
        allowed_functions: list[Function],
        prompt: str) -> dict[str, object]:
    """Translate one prompt into a structured function call.

    Args:
        llm: SDK model used for constrained token selection.
        brainwash: System instructions, including the available functions.
        allowed_functions: Validated functions the model may select.
        prompt: Original natural-language request.

    Returns:
        An object containing prompt, name, and parameters.

    Raises:
        ValueError: No valid selection is possible or the result is invalid
            JSON.
    """

    # adding brainwash as system message:
    serialized_prompt: t_token_ids = format_token_ids_mssg([{
        "role": ROLE_IDS["system"],
        "content": llm.encode(brainwash)[0].tolist()}])

    # adding user message
    user_token_ids = format_token_ids_mssg(
        [
            {
                "role": ROLE_IDS["user"],
                "content": llm.encode(prompt)[0].tolist()
            }
        ]
    )
    serialized_prompt.extend(user_token_ids)

    # appending the assistant prefix
    serialized_prompt.extend(prefix_assistant_token_ids())
    output_start = len(serialized_prompt)

    # creating the output prefix
    prefix_of_output: t_token_ids = llm.encode(
        f'{{"prompt": {json.dumps(prompt)}, "name": "'
    )[0].tolist()

    serialized_prompt.extend(prefix_of_output)

    # generating function name
    allowed_func_names_token_ids: dict[str, t_token_ids] = {
        func.name: llm.encode(json.dumps(func.name)[1:-1])[0].tolist()
        for func in allowed_functions
    }
    visualization.show("Step", "Choosing a function")
    generated_fn_name: t_token_ids = generate_function_name(
        llm,
        serialized_prompt,
        allowed_func_names_token_ids,
    )
    serialized_prompt.extend(generated_fn_name)
    selected_function = next(
        func for func in allowed_functions
        if allowed_func_names_token_ids[func.name] == generated_fn_name
    )
    visualization.show("Function", selected_function.name)

    # appending the parameters prefix
    serialized_prompt.extend(llm.encode('", "parameters": ')[0].tolist())

    # generating parameters
    generated_parameters: t_token_ids = generate_parameters(
        llm,
        serialized_prompt,
        allowed_parameters=selected_function.parameters,
    )
    serialized_prompt.extend(generated_parameters)

    # closing the JSON object
    serialized_prompt.extend(llm.encode("}")[0].tolist())

    result: dict[str, object] = json.loads(
        llm.decode(serialized_prompt[output_start:])
    )
    return result


def main() -> None:
    """Load inputs, generate function calls, and save one JSON array.

    Calls are also printed to standard output. Identical prompts reuse their
    earlier result within this run.

    Raises:
        SystemExit: Arguments or input files are invalid.
        ValueError: Generation cannot produce a valid function call.
        OSError: The output directory or file cannot be written.
    """
    # loading command line arguments
    args = init_arg_parser()
    visualization.enabled = args.visualize

    try:
        brainwash = load_file("brainwash.txt")
    except Exception as e:
        write_error("Error", f"Failed to load brainwash file: {e}")
        sys.exit(1)

    # loading input prompts
    try:
        input_prompts: list[Prompt] = load_prompts(args.input)
    except Exception as e:
        write_error("Error", f"Failed to load input prompts: {e}")
        sys.exit(1)

    # loading input functions
    try:
        input_functions = load_functions(args.functions_definition)
    except Exception as e:
        write_error("Error", f"Failed to load input functions: {e}")
        sys.exit(1)

    # adding functions to the brainwash
    for function in input_functions:
        brainwash += "\n" + function.dump_json() + "\n"

    llm: Small_LLM_Model | None = None
    generated_results: list[dict[str, object]] = []
    prompt_cache: dict[str, dict[str, object]] = {}
    output_file = Path(args.output)

    for p in input_prompts:
        visualization.show("Prompt", p.prompt)
        # Generate once per exact prompt, then reuse the saved result.
        if p.prompt not in prompt_cache:
            if llm is None:
                llm = Small_LLM_Model()
            prompt_cache[p.prompt] = constrained_decoding(
                llm, brainwash, input_functions, p.prompt,
            )
        else:
            visualization.show("Cache", "Reusing the saved result")
        func = prompt_cache[p.prompt]
        generated_results.append(func)
        visualization.show("Done", "Function call ready")
        if not args.visualize:
            print(json.dumps(func, ensure_ascii=False), flush=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            generated_results, f, indent=4,
            ensure_ascii=False, allow_nan=False,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_error("Error", str(error))
        sys.exit(1)
