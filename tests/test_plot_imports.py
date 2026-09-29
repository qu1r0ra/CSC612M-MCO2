"""Importing analysis modules must not select a plotting backend."""

import subprocess
import sys


def test_stoquant_imports_do_not_load_pyplot():
    code = """
import importlib
import sys
for name in ('report', 'publication', 'k1_ab', 'unbiasedness'):
    importlib.import_module('stoquant.' + name)
assert 'matplotlib.pyplot' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
