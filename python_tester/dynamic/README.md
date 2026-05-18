# Dynamic-GICP tests

Synthetic indoor-scene tests for the dynamic outlier rejection / IRLS-GNC
feature on the `feature/dynamic-outlier-rejection` branch.

## Running

```bash
# from submodules/fast_gicp
cmake --build build_dyn -j$(nproc)        # build (no install)
PYTHONPATH=build_dyn pytest -v python_tester/dynamic
```

Each test file loads `pygicp` from `build_dyn/` via `_loader.py` and asserts
that the resolved module path lives inside `build_dyn/`. If the nsl2-installed
`pygicp` is loaded by mistake, the assertion in `_loader.py` fails immediately.

Per-test JSON metrics are written under `python_tester/dynamic/results/` for
post-hoc inspection.

## Scenes

| File | Scene(s) | What it tests |
|------|----------|---------------|
| `test_dynamic_static.py` | S10, S5 | static regression + degenerate planar safety |
| `test_dynamic_movers.py` | S1, S2, S3 | dynamic outlier rejection; S3 (corridor) is the hard case |
| `test_dynamic_anchors.py` | S8 | sparse 3D anchors with bad anchors on dynamics |
| `test_basin_fragility.py` | S4, S6, S7 | mirror-symmetric, large init offset, combined worst case |
| `test_integration_2dgs_surfel.py` | S9 + unsupported paths | 2DGS surfel path; VGICP/CUDA error path |
| `test_performance.py` | S10 | wall-clock overhead vs baseline |

Scene primitives and factories live in `synthetic_indoor.py`.
