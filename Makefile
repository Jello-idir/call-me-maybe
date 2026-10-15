
install:
	@uv sync

run:
	@uv run src

clean:
	@rm -rf __pycache__
