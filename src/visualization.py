"""Optional terminal display for the steps of constrained decoding."""

import json
import sys

enabled = False


def show(stage: str, text: str) -> None:
    """Print one generation step when the display is enabled.

    Args:
        stage: Short label such as Prompt, Name, Value, or Cache.
        text: Current text at that step. Control characters are escaped
            to keep each update on a single line.
    """
    if not enabled:
        return

    color = "\033[1;34m" if stage == "Prompt" else "\033[33m"
    value = json.dumps(text, ensure_ascii=False)

    if stage == "Prompt":
        print(file=sys.stderr, flush=True)
        print(
            f"{color}{stage:>10} | {value}\033[0m",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"{color}{stage:>10} | \033[0m{value}",
            file=sys.stderr,
            flush=True,
        )
