"""Path/viz wiring for the CUDA tests.

Reuses the shared synthetic_indoor + viz modules from python_tester/dynamic/
without duplicating them, while keeping CUDA tests isolated under their own
results directory so the CPU + CUDA gallery don't overwrite each other.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, "..", "dynamic"))
for path in (HERE, SHARED):
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_sessionfinish(session, exitstatus):
    try:
        import viz as _viz
        results = os.path.join(HERE, "results")
        if os.path.isdir(results):
            _viz.write_index_html(results)
    except Exception as e:
        print(f"[conftest cuda] viz aggregation skipped: {e}")
