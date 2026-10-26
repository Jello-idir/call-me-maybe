*This project has been created as part of the 42 curriculum by `<login>`.*

# Call Me Maybe

## Description

Convert natural-language requests into JSON function calls using
Qwen/Qwen3-0.6B and the supplied `llm_sdk`. The program selects a function and
extracts its arguments; it does not execute the function.

## Instructions

Requires Python 3.10 or later and `uv`. Run from the project root:

```bash
uv sync
uv run python -m src
```

The first run downloads the model. Default paths can be changed:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Example result for “What is the sum of 2 and 3?”:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
```

Results are saved as one JSON array. Keep generated output out of Git.
The Makefile provides `install`, `run`, `debug`, `clean`, `lint`, and
`lint-strict` targets.

## Algorithm

1. Validate prompts and function definitions with Pydantic. Add the definitions
   to the system prompt and format the conversation for Qwen.
2. Choose the highest-scoring allowed token at each step of the function name.
   Each choice must match the prefix of a name from the definitions.
3. Choose each required parameter name once, then generate its typed value.
4. Check value candidates before accepting them. Boolean prefixes must match
   `true` or `false`. For unfinished numbers and strings, try temporary endings
   and validate with `json.loads()`. These endings never enter the output.
5. Stop values at commas or closing braces outside quoted strings. Reject
   extra text after the separator, assemble the call, and save the results.

## Design decisions

Use the SDK's public methods, Python's JSON parser, and explicit JSON structure.
Supported values are strings, finite numbers, and booleans. Generation ends at
a valid separator, with no fixed token limit.

## Challenges

Overlapping function names are distinguished by their closing quote and comma.
Decoding the full value prefix handles Unicode characters split across tokens.
Tracking quotes and escapes prevents commas inside strings from ending values.

## Testing

During development, 15 temporary tests checked invalid token rejection,
Unicode, escaping, overlapping names, long values, and output files.
The test suite was removed from the submission.

For a manual check, run `make run`, compare the selected functions and arguments
with the input prompts, and verify that the output parses:

```bash
uv run python -m json.tool data/output/function_calls.json
```

## Performance analysis

All 15 development tests passed before removal. Full-model accuracy and the subject's
five-minute runtime target have not been verified. Repeated model calls and
sorting vocabulary scores make longer outputs slower. Syntax checks do not
guarantee that the model chooses the semantically correct function or arguments.

## Resources

- [Qwen3-0.6B model documentation](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [Google Python docstring style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

AI assistance was used for constrained decoding, JSON validation, tests,
Makefile updates, code explanations, docstrings, and this README.
