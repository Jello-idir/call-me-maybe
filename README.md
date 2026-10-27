*This project has been created as part of the 42 curriculum by aait-idi.*

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
The Makefile provides `install`, `run`, `run-visualize`, `debug`, `clean`, `lint`, and
`lint-strict` targets.

The application has three Python files next to each other:

- `src/__main__.py`: command-line handling and constrained decoding.
- `src/models.py`: Pydantic schemas and Qwen token constants.
- `src/visualization.py`: optional terminal display.

## Bonus: caching repeated prompts

The performance optimization bonus uses a dictionary called `prompt_cache`
in `main()`. The key is the exact prompt text; the value is its generated
function call. The first occurrence uses the model and saves the result.
Later occurrences reuse that result, skipping generation.

Every input prompt still gets an output entry in the original order. The cache
starts empty each run, so changing function definitions between runs is safe.
Only identical text matches: different capitalization or spacing is a new
prompt. Inputs with no repetitions get no speedup from this bonus.

Try the example with three prompts, where the first and third are identical:

```bash
uv run python -m src \
  --input data/input/function_calling_cache_demo.json \
  --output data/output/cache_demo.json
```

The output contains three calls, but generation runs only twice.

## Bonus: generation visualization

Add `--visualize` to see the generation steps in the terminal:

```bash
uv run python -m src --visualize \
  --input data/input/function_calling_cache_demo.json \
  --output data/output/cache_demo.json
```

The display shows the prompt, names as their tokens are accepted, the chosen
function, and parameter values as they grow. Rejected token candidates are
not displayed. A repeated prompt shows a `Cache` message instead of generating
again. Each line is a snapshot of the text so far, including JSON punctuation.

For example, a value could grow like this (token boundaries can vary):

```text
 Parameter | "name (string)"
     Value | "\""
     Value | "\"john"
     Value | "\"john\"}"
      Done | "Function call ready"
```

The display uses standard error. With visualization enabled, progress replaces
the usual result printing on standard output; the saved JSON has the same
format in both modes. Visualization is off by default and needs no extra
libraries.

All display formatting lives in `src/visualization.py`. It has one flag,
`enabled`, and one function, `show(stage, text)`. `main()` sets the flag, and
the decoder calls `show()` after accepting tokens or completing a step.
To disable it, leave out `--visualize`. To remove it completely, remove the
`visualization` import and calls/guards from `src/__main__.py`, its flag setup
and `--visualize` argument, then delete `src/visualization.py`. The cache's
`else` branch can also be removed because it only displays a message.

## Algorithm

1. Validate prompts and function definitions with Pydantic. Add the definitions
   to the system prompt and format the conversation for Qwen.
2. Choose the highest-scoring allowed token at each step of the function name.
   Each choice must match the prefix of a name from the definitions. When only
   one token is possible, append it directly without a model call. The model
   still decides whenever more than one valid token is available.
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

All application data classes use Pydantic for validation. Token constants are
plain dictionaries. Empty function lists, duplicate function names, unsupported
types, and malformed inputs are rejected before loading the model. An empty
prompt list writes `[]` without loading the model.

Function definitions use compact JSON to reduce the context processed at every
generation step. No SDK internals or model-specific heuristics select functions.
`brainwash.txt` gives general extraction instructions and regex/symbol syntax
reminders; function definitions still come from the input file, and the model
chooses all function names and parameter values.

## Challenges

Overlapping function names are distinguished by their closing quote and comma.
Decoding the full value prefix handles Unicode characters split across tokens.
Tracking quotes and escapes prevents commas inside strings from ending values.

## Testing

Before submission, 17 temporary tests checked input validation, error handling,
typed values, Unicode and escaping, overlapping and custom names, functions
without parameters, cache reuse, visualization, and output files. All passed.
These checks are kept outside the repository, as requested by the subject.

The caching demo was checked with the local Qwen3-0.6B model: three output
entries, two generation calls, correct arguments, and matching repeated results.

The same demo with `--visualize` produced matching JSON and showed the two
generations plus one cache hit. Temporary checks also verified that disabling
the display is silent, rejected tokens stay hidden, and control characters
are escaped.

Run `make lint` for the required flake8 and mypy checks, or `make lint-strict`
for stricter typing. The configuration checks the application and excludes the
supplied third-party SDK. Mypy uses `typings/llm_sdk/__init__.pyi` for the SDK's
public API, avoiding a duplicate-module error from its source and stub.

For a manual check, run `make run`, compare the selected functions and arguments
with the input prompts, and verify that the output parses:

```bash
uv run python -m json.tool data/output/function_calls.json
```

## Performance analysis

The final check processed the 11 supplied prompts in 149 seconds on the local
CPU with default thread settings and an already-cached Qwen3-0.6B model.
All 11 results were valid JSON with the required keys and parameter types.
Function selection was correct for all 11; complete argument extraction was
correct for 10 of 11 (90.9%), meeting the subject's stated accuracy and runtime
targets on this sample.

Known limitation: the vowel-replacement request produced `"regex": "aeiou"`
and `"replacement": " *"`, which do not perform the requested replacement.
Constrained decoding guarantees syntax and types, not the meaning of a regex
or the correctness of extracted arguments. Accuracy on unseen prompts and
function sets, and runtime on other hardware, may differ.

Skipping model calls for forced tokens and using compact definitions reduce
runtime. Prompt caching saves additional work when requests repeat. The SDK
still processes the full context for each prediction, so longer values take
more time. All 17 temporary checks, flake8, and strict mypy passed.

## Resources

- [Qwen3-0.6B model documentation](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)

AI assistance was used to **HELP** in constrained decoding, JSON validation,
prompt caching, and was used to **DO** this README, docstrings, generation
visualization, automated checks, the models file move, and submission fixes.
