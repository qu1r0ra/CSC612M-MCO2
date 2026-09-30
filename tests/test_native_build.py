from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from stoquant.native_build import command_plan, source_lists

ROOT = Path(__file__).resolve().parents[1]
NATIVE_LINT_SPEC = spec_from_file_location(
    "native_lint_under_test", ROOT / "tools" / "native_lint.py"
)
if NATIVE_LINT_SPEC is None or NATIVE_LINT_SPEC.loader is None:
    raise RuntimeError("cannot load native_lint module")
NATIVE_LINT = module_from_spec(NATIVE_LINT_SPEC)
NATIVE_LINT_SPEC.loader.exec_module(NATIVE_LINT)
c_units = NATIVE_LINT.c_units


def test_source_list_addition_reaches_the_build_stamp_plan_and_native_lint(tmp_path):
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    justfile = justfile.replace(
        'native_host_sources := "native/main.c native/cli.c',
        'native_host_sources := "native/extra_unit.c native/main.c native/cli.c',
        1,
    )
    (tmp_path / "justfile").write_text(justfile, encoding="utf-8")

    sources = source_lists(tmp_path)
    commands, _ = command_plan(tmp_path, "build-cpu")
    lint_sources = [source for source, _ in c_units(tmp_path)]

    assert sources["native_host_sources"][0] == "native/extra_unit.c"
    assert sum("native/extra_unit.c" in command for command in commands) == 1
    assert lint_sources.count("native/extra_unit.c") == 2


def test_native_build_plans_include_each_registered_test_source():
    sources = source_lists(ROOT)
    expected_by_recipe = {
        "build-codec-test": "test_codec_sources",
        "build-quantizer-test": "test_quantizer_sources",
        "build-avx2-test": "test_avx2_sources",
        "build-rng-cpu-test": "test_rng_cpu_sources",
    }
    for recipe, variable in expected_by_recipe.items():
        commands, _ = command_plan(ROOT, recipe)
        command_sources = {token for command in commands for token in command}
        assert set(sources[variable]) <= command_sources


def test_gcc_ubsan_plan_builds_cli_and_all_native_c_tests():
    commands, binary = command_plan(ROOT, "test-cpu-gcc-ubsan")
    command_text = " ".join(token for command in commands for token in command)

    assert binary == ""
    assert "build/stoquant_gcc_ubsan" in command_text
    assert "-fsanitize=undefined" in command_text
    assert "-fsanitize=undefined" in commands[-1]
    for source_variable in (
        "test_codec_sources",
        "test_quantizer_sources",
        "test_avx2_sources",
        "test_rng_cpu_sources",
    ):
        for source in sources_for(source_variable):
            assert source in command_text


def sources_for(variable: str) -> list[str]:
    return source_lists(ROOT)[variable]
