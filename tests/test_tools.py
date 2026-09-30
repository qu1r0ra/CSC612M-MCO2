import subprocess
import sys

from stoquant import layout
from stoquant.native_build import BUILD_RECIPES, recipe_commands


def test_fault_injection_define_is_only_in_the_fault_recipe():
    define = "SQ_CUDA_FAULT_INJECTION"
    users = {
        recipe
        for recipe in BUILD_RECIPES
        if define in "\n".join(recipe_commands(layout.ROOT, recipe))
    }
    assert users == {"build-cuda-fault"}
    text = (layout.ROOT / "justfile").read_text(encoding="utf-8")
    assert define not in text.split("build-cuda-fault")[0]


def test_equivalence_compare_resolves_a_baseline_by_commit_prefix(tmp_path):
    (tmp_path / "abc1234.json").write_text('{"format": -1}', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(layout.TOOLS_DIR / "equivalence.py"),
            "compare",
            "--store",
            str(tmp_path),
            "--baseline",
            "abc12",
        ],
        cwd=layout.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "abc1234.json has format -1" in result.stderr, result.stderr
