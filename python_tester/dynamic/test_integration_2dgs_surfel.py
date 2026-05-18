"""S9: 2DGS surfel covariance path with dynamic rejection enabled.

Validates that the surfel-covariance entry point still works when rejection is
active and that the diagnostics are populated.
"""

import json
import os

import numpy as np
import pytest

from _loader import load, results_dir
import synthetic_indoor as si
import viz


@pytest.fixture(scope="module")
def pg():
    return load()


def test_S9_2dgs_surfel_with_rejection(pg):
    base, src_rot, src_scl, tgt_rot, tgt_scl = si.scene_S9_2dgs_surfels()

    g = pg.FastGICP()
    g.set_num_threads(8)
    g.set_max_correspondence_distance(1.0)
    g.set_input_target(base.target)
    g.set_input_source(base.source)
    g.set_source_covariances_from_2dgs(src_rot.astype(np.float32),
                                        src_scl.astype(np.float32),
                                        "physical", 0.05, 1e-3)
    g.set_target_covariances_from_2dgs(tgt_rot.astype(np.float32),
                                        tgt_scl.astype(np.float32),
                                        "physical", 0.05, 1e-3)
    g.set_dynamic_rejection_config({"enable": True, "kernel": "GEMAN_MCCLURE",
                                    "warmup_iterations": 3, "min_inlier_ratio": 0.2})
    T_est = g.align(np.eye(4, dtype=np.float32)).astype(np.float64)
    rot, tr = si.pose_error(base.gt_pose, T_est)
    diag = g.get_dynamic_rejection_diagnostics()
    weights = np.asarray(g.get_dynamic_correspondence_weights())

    out = dict(scene="S9_2dgs_surfel",
               rot_deg=rot, trans_m=tr, diagnostics=diag,
               weights_size=int(weights.size),
               inlier_count=int((weights >= 0.5).sum()))
    with open(os.path.join(results_dir(), "S9.json"), "w") as f:
        json.dump(out, f, indent=2)
    viz.render_full_scene(
        results_dir(), "S9",
        scene_desc=f"S9: {base.description} (2DGS surfel covariance)",
        source=base.source, target=base.target, gt_pose=base.gt_pose,
        est_pose_dynrej=T_est,
        source_is_dynamic=base.source_is_dynamic,
        weights=weights,
        residuals=np.asarray(g.get_dynamic_correspondence_residuals()),
        diagnostics=diag,
        rot_err_dynrej=rot, trans_err_dynrej=tr)

    assert diag["enabled"] is True
    assert weights.size == base.source.shape[0]
    # Pose must be reasonably good.
    assert rot < 2.0, f"rot {rot:.3f} deg"
    assert tr < 0.05, f"trans {tr:.4f} m"


def test_unsupported_paths_throw(pg):
    """VGICP / NDT / VGICP_CUDA must raise when dynamic rejection is enabled."""
    v = pg.FastVGICP()
    with pytest.raises(RuntimeError):
        v.set_dynamic_rejection_config({"enable": True, "kernel": "GEMAN_MCCLURE"})

    # Equivalent setter path:
    v2 = pg.FastVGICP()
    with pytest.raises(RuntimeError):
        v2.set_dynamic_rejection_enabled(True)
