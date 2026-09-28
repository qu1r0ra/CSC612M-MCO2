"""Repository locations shared by the package, tools and tests.

ROOT assumes the editable install that `uv sync` creates, so this file still
lives at src/stoquant/layout.py inside the checkout.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "build"
NATIVE_DIR = ROOT / "native"
TOOLS_DIR = ROOT / "tools"
