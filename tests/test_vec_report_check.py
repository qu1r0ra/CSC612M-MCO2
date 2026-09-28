from stoquant import vectorization as vec_report_check

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
    report = r"""native\quantizer_avx2.c(3) : info C5001: loop vectorized
native\quantizer_avx2.c(6) : info C5002: loop not vectorized due to reason '1203'
native\quantizer_avx2.c(7) : info C5002: loop not vectorized due to reason '1200'
"""

    assert vec_report_check.hot_lines(SOURCE) == [3, 7]
    assert vec_report_check.unvectorized_hot_loops(SOURCE, report) == [7]


def test_a_loop_vectorized_in_every_inlined_copy_passes():
    report = r"""native\quantizer_avx2.c(7) : info C5001: loop vectorized
native\quantizer_avx2.c(3) : info C5001: loop vectorized
native\quantizer_avx2.c(3) : info C5001: loop vectorized
"""

    assert vec_report_check.unvectorized_hot_loops(SOURCE, report) == []


def test_a_scalar_inlined_copy_fails_the_tagged_loop():
    report = r"""native\quantizer_avx2.c(7) : info C5001: loop vectorized
native\quantizer_avx2.c(3) : info C5001: loop vectorized
native\quantizer_avx2.c(3) : info C5002: loop not vectorized due to reason '1200'
"""

    assert vec_report_check.unvectorized_hot_loops(SOURCE, report) == [3]


def test_other_files_do_not_satisfy_the_check():
    report = r"""native\other.c(3) : info C5001: loop vectorized
tests\test_quantizer_avx2.c(7) : info C5001: loop vectorized
"""

    assert vec_report_check.unvectorized_hot_loops(SOURCE, report) == [3, 7]
