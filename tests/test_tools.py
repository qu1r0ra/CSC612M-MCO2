import subprocess
import sys

import pytest

from stoquant import layout


@pytest.mark.parametrize("tool", ["diag_resident", "pin_resident", "trace_resident"])
def test_residency_tool_imports_the_installed_package(tool):
    result = subprocess.run(
        [sys.executable, str(layout.TOOLS_DIR / f"{tool}.py"), "--help"],
        cwd=layout.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
