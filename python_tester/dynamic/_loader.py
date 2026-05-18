"""Shared loader that imports the locally-built pygicp from build_dyn/.

Asserts that the loaded module is from build_dyn/ — sentinel against the
preinstalled pygicp in the active conda env getting loaded by mistake.
"""

import os
import sys


def load():
    here = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.abspath(os.path.join(here, "..", "..", "build_dyn"))
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import pygicp
    assert "build_dyn" in pygicp.__file__, (
        f"wrong pygicp loaded ({pygicp.__file__}); expected build_dyn/. "
        "This means the conda-env-installed pygicp was loaded — install isolation broken."
    )
    return pygicp


def results_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results")
    os.makedirs(out, exist_ok=True)
    return out
