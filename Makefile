.PHONY: install voca run debug clean lint lint-strict

install:
	@uv sync

voca:
	@cat "$$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/vocab.json | jq

run:
	@uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calls.json

debug:
	@uv run python -m pdb -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calls.json

clean:
	@find . -type d \( -name .venv -o -name .git \) -prune -o \
	-type d \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) \
	-prune -exec rm -rf {} +
	@find . -type d \( -name .venv -o -name .git \) -prune -o \
	-type f \( -name '*.pyc' -o -name '*.pyo' \) -exec rm -f {} +

lint:
	@uv run --with flake8 flake8 . --jobs 1 --extend-exclude .venv
	@uv run --with mypy mypy . --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@uv run --with flake8 flake8 . --jobs 1 --extend-exclude .venv
	@uv run --with mypy mypy . --strict
