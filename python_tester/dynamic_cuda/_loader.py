"""Loader for the CUDA-built pygicp (from build_dyn_cuda/)."""

import os
import sys


def load():
    here = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.abspath(os.path.join(here, "..", "..", "build_dyn_cuda"))
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import pygicp
    assert "build_dyn_cuda" in pygicp.__file__, (
        f"wrong pygicp loaded ({pygicp.__file__}); expected build_dyn_cuda/. "
        "This means the conda-env-installed pygicp or the CPU-only build was "
        "loaded — CUDA build isolation broken.")
    assert hasattr(pygicp, "FastGICPCuda"), (
        "pygicp.FastGICPCuda not exposed — CUDA path was not compiled in.")
    return pygicp


def results_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results")
    os.makedirs(out, exist_ok=True)
    return out
