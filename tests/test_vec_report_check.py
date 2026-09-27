import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vec_report_check.py"
spec = importlib.util.spec_from_file_location("vec_report_check", SCRIPT)
vec_report_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vec_report_check)

SOURCE = """int f(void)
{
    for (i = 0; i < n; i++) /* avx2-hot */
        a[i] = b[i];
    for (i = 0; i < n; i++)
        c[i] = d[2 * i];
    for (j = 0; j < n; j++) { /* avx2-hot */
    }
}
"""


def test_every_tagged_loop_must_be_reported_vectorized():
    report = r"""src\quantizer_avx2.c(3) : info C5001: loop vectorized
src\quantizer_avx2.c(6) : info C5002: loop not vectorized due to reason '1203'
src\quantizer_avx2.c(7) : info C5002: loop not vectorized due to reason '1200'
"""

    assert vec_report_check.hot_lines(SOURCE) == [3, 7]
    assert vec_report_check.unvectorized_hot_loops(SOURCE, report) == [7]


def test_a_loop_vectorized_in_any_inlined_copy_counts_once():
    report = r"""src\quantizer_avx2.c(7) : info C5001: loop vectorized
src\quantizer_avx2.c(3) : info C5001: loop vectorized
src\quantizer_avx2.c(3) : info C5001: loop vectorized
"""

    assert vec_report_check.unvectorized_hot_loops(SOURCE, report) == []


def test_other_files_do_not_satisfy_the_check():
    report = r"""src\other.c(3) : info C5001: loop vectorized
tests\test_quantizer_avx2.c(7) : info C5001: loop vectorized
"""

    assert vec_report_check.unvectorized_hot_loops(SOURCE, report) == [3, 7]
