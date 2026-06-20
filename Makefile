
install:
	uv sync

run:
	uv run src

clean:
	rm -rf .venv
	rm -rf __pycache__
