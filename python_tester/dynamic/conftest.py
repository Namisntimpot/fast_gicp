"""pytest fixtures for dynamic-GICP tests.

Adds the local module directory to sys.path so individual test files can do
``from _loader import load`` and ``import synthetic_indoor``.

Also installs a session-scoped finalizer that writes ``results/index.html``
aggregating any PNG/JSON the individual tests produced.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def pytest_sessionfinish(session, exitstatus):  # noqa: D401
    """After all tests run, rebuild results/index.html from whatever files
    landed under results/. Safe even if no viz was produced."""
    try:
        import viz as _viz  # noqa: WPS433
        results = os.path.join(HERE, "results")
        if os.path.isdir(results):
            _viz.write_index_html(results)
    except Exception as e:
        print(f"[conftest] viz aggregation skipped: {e}")
