
install:
	@uv sync


voca:
	cat /home/aixel/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json | jq

run:
	@uv run src
# 	uv run python -m src \
# 	--functions_definition data/input/functions_definition.json \
# 	--input data/input/function_calling_tests.json \
# 	--output data/output/function_calls.json

clean:
	@find . -name "__pycache__" -type d -exec rm -rf {} +
	@find . -name "*.pyc" -type f -delete
