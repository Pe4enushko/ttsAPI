"""
Entry point that supports an extra --flashattn flag before forwarding all
remaining arguments to uvicorn.

Usage examples:
    python run.py                          # start without flash attention
    python run.py --flashattn              # start with flash_attention_2
    python run.py --flashattn --port 8080  # flash attention + custom port
"""

import os
import sys
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--flashattn", action="store_true", default=False)
known, uvicorn_args = parser.parse_known_args()

if known.flashattn:
    os.environ["USE_FLASH_ATTN"] = "1"

import uvicorn  # noqa: E402  (import after env var is set)

# Build uvicorn CLI args; default to "main:app" if not provided
if not any(a.startswith("main") or ":" in a for a in uvicorn_args):
    uvicorn_args = ["main:app"] + uvicorn_args

sys.argv = [sys.argv[0]] + uvicorn_args
uvicorn.main()
