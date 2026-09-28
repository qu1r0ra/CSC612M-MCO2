"""CUDA test gating: `cuda`-marked tests run only with STOQUANT_TEST_CUDA=1.

`--require-cuda` turns a run in which no `cuda` test executed into a failure, so
`just test-cuda` cannot pass by skipping everything.
"""

import os

import pytest

CUDA_ENABLED = os.environ.get("STOQUANT_TEST_CUDA") == "1"
CUDA_TESTS_RUN = pytest.StashKey[int]()


def pytest_addoption(parser):
    parser.addoption(
        "--require-cuda",
        action="store_true",
        help="fail the session unless at least one cuda-marked test executed",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs a CUDA device and STOQUANT_TEST_CUDA=1")
    config.stash[CUDA_TESTS_RUN] = 0


def pytest_collection_modifyitems(config, items):
    if CUDA_ENABLED:
        return
    skip = pytest.mark.skip(reason="run with `just test-cuda` on a CUDA device")
    for item in items:
        if item.get_closest_marker("cuda"):
            item.add_marker(skip)


def pytest_runtest_call(item):
    if item.get_closest_marker("cuda"):
        item.config.stash[CUDA_TESTS_RUN] += 1


def pytest_sessionfinish(session):
    config = session.config
    passing = session.exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED)
    if passing and config.getoption("--require-cuda") and config.stash[CUDA_TESTS_RUN] == 0:
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.ensure_newline()
            reporter.write_line("--require-cuda: no CUDA test executed", red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
