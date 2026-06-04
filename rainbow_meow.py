#!/usr/bin/env python3
"""Print a rainbow-colored `мяу мяу` to the terminal."""

from __future__ import annotations

import argparse


RAINBOW_COLORS = [
    "\033[31m",  # red
    "\033[33m",  # yellow
    "\033[32m",  # green
    "\033[36m",  # cyan
    "\033[34m",  # blue
    "\033[35m",  # magenta
]
RESET = "\033[0m"


def rainbow_text(text: str) -> str:
    colored_chars = []
    color_index = 0

    for char in text:
        if char.isspace():
            colored_chars.append(char)
            continue

        colored_chars.append(f"{RAINBOW_COLORS[color_index % len(RAINBOW_COLORS)]}{char}")
        color_index += 1

    return "".join(colored_chars) + RESET


def main() -> None:
    parser = argparse.ArgumentParser(description="Print rainbow text.")
    parser.add_argument("text", nargs="?", default="мяу мяу", help="Text to colorize")
    args = parser.parse_args()

    print(rainbow_text(args.text))


if __name__ == "__main__":
    main()
