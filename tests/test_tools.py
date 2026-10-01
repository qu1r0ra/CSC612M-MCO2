import importlib.util
import subprocess
import sys

from stoquant import layout
from stoquant.native_build import BUILD_RECIPES, recipe_commands

equivalence_spec = importlib.util.spec_from_file_location(
    "equivalence", layout.TOOLS_DIR / "equivalence.py"
)
assert equivalence_spec is not None and equivalence_spec.loader is not None
equivalence = importlib.util.module_from_spec(equivalence_spec)
equivalence_spec.loader.exec_module(equivalence)
flatten_fingerprint = equivalence.flatten_fingerprint


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


def test_equivalence_compare_redacts_build_stamp_revision_and_binary_hash():
    def fingerprint(revision, binary_hash):
        return {
            "live": {
                "bench-dense": {
                    "files": {
                        "manifest.json": {
                            "build_stamp/revision": revision,
                            "build_stamp/binary_sha256": binary_hash,
                            "build_stamp/tree_dirty": False,
                            "build_stamp/tree_fingerprint": "clean-tree",
                        }
                    }
                }
            }
        }

    captured = flatten_fingerprint(fingerprint("bba8573", "captured-hash"))
    rebuilt = flatten_fingerprint(fingerprint("38b23e9", "rebuilt-hash"))

    assert captured == rebuilt
    assert captured["live/bench-dense/files/manifest.json/build_stamp/tree_dirty"] is False
    assert (
        captured["live/bench-dense/files/manifest.json/build_stamp/tree_fingerprint"]
        == "clean-tree"
    )
