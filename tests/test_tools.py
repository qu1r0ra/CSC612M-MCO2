import subprocess
import sys

from stoquant import layout


def _recipe_bodies() -> dict[str, list[str]]:
    bodies: dict[str, list[str]] = {}
    name = ""
    for line in (layout.ROOT / "justfile").read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace() and not line.startswith(("#", "[")):
            name = line.split(":")[0].split()[0]
            bodies.setdefault(name, [])
        elif name and line.startswith((" ", "\t")):
            bodies[name].append(line)
    return bodies


def test_fault_injection_define_is_only_in_the_fault_recipe():
    define = "SQ_CUDA_FAULT_INJECTION"
    users = {name for name, lines in _recipe_bodies().items() if define in "\n".join(lines)}
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
