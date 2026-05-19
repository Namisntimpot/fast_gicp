+ 抑制pcl的警告，不再频繁输出`fastgicp Assignment with new_width equal to 0,setting width to size of the cloud and height to 1`.
+ cmake中必须加 `-DBUILD_apps=OFF`，摆脱对visualization相关包的依赖。CMakeList.txt中，改为了`find_package(PCL REQUIRED COMPONENTS common io filters search registration)`  
+ 要手动安装pybinding11，没有用third_party里的那些, CMakeList里：`find_package(pybind11 CONFIG REQUIRED)`  
+ `setup.py` 中，如果你的系统依赖包不是装在conda环境，就注释掉`59-64`行.


* Ref: https://github.com/SMRT-AIST/fast_gicp

- Add some useful functions for both cpp & python
- Modify gicp as it utilizes raw covariance by following normalized_ellipse mode (not plane mode), in order to meet the scales for multiple 3D pointclouds
=> scale = scale / scale[1] .max(1e-3)
  
* note that cov = R*S*(R*S)^T = R*SS*R^T,   S = scale.asDiagonal();
* here, R = quaternion.toRotation();
* q = (q_x, q_y, q_z, q_w)
* R and SS can be obtained by SVD; R=U, scale**2 = singular_values.array()

## Optimization summary

The base geometry term is still the standard GICP objective

$$
E_{\text{gicp}}(T)=\sum_i \mathbf{e}_i(T)^\top \mathbf{\Lambda}_i(T)\mathbf{e}_i(T)
$$

with

$$
\mathbf{e}_i(T)=\mathbf{p}_i^t-T\mathbf{p}_i^s
$$

and

$$
\mathbf{\Lambda}_i(T)=\left(\mathbf{C}_i^t+T\mathbf{C}_i^sT^\top\right)^{-1}.
$$

Here $\mathbf{p}_i^s,\mathbf{p}_i^t \in \mathbb{R}^3$ are the matched source/target points and $\mathbf{C}_i^s,\mathbf{C}_i^t$ are their local covariances.

### Sparse 3D anchor term

If external visual frontends provide matched 3D-3D pairs, the optimizer adds a sparse anchor term

$$
E_{\text{anchor}}(T)=\sum_k \frac{w_k}{\sigma_k^2}\left\lVert \mathbf{q}_k^t-T\mathbf{q}_k^s \right\rVert_2^2
$$

where $\mathbf{q}_k^s,\mathbf{q}_k^t$ are anchor pairs, $w_k$ is an optional confidence, and $\sigma_k$ is an optional scale.

The implementation first builds this raw anchor term and then multiplies it by an effective global scale

$$
\lambda_{\text{anchor}}^{\text{eff}}=\lambda_{\text{anchor}}^{\text{user}}\gamma_{\text{anchor}}
$$

where `anchor_objective_weight` is $\lambda_{\text{anchor}}^{\text{user}}$ and `balance_mode` controls $\gamma_{\text{anchor}}$:

- `NONE`: $\gamma_{\text{anchor}} = 1$
- `BY_COUNT`: balance by geometric residual count versus anchor count
- `BY_HESSIAN_TRACE`: balance by unlocked Hessian trace of the geometry term versus the anchor term

The default on this branch is `BY_HESSIAN_TRACE`.

### Automatic soft suppression for ambiguous DOFs

When observability check is enabled, the code eigen-decomposes the 6x6 Hessian and estimates ambiguous directions from its small eigenvalues. A diagonal soft prior is then added only on the selected ambiguous DOFs:

$$
E_{\text{soft}}(\boldsymbol{\delta})=\boldsymbol{\delta}^\top \mathbf{W}\boldsymbol{\delta}
$$

where $\boldsymbol{\delta}\in\mathfrak{se}(3)$ is the current incremental update and $\mathbf{W}$ is diagonal. In the implementation,

$$
W_{jj}=\alpha \lambda_{\max} a_j
$$

for automatically suppressed DOF $j$, where $\alpha$ is `auto_soft_prior_strength`, $\lambda_{\max}$ is the largest Hessian eigenvalue, and $a_j$ is the ambiguity score of that DOF.

### Smooth observability prior (alternative to binary suppression)

The binary soft suppression above suffers from two problems: (1) the threshold creates all-or-nothing behavior—a DOF is either fully degenerate or fully unconstrained; (2) the regularization scales with $\lambda_{\max}$ of the **combined** Hessian, which can be very large and over-constrain even mildly degenerate directions.

When `use_smooth_prior = true`, a smooth sigmoid-based prior replaces the binary selection. The key idea is to analyze the **geometry** Hessian alone (when `analyze_geometry_separately = true`) to detect what pure GICP cannot resolve, then apply a smooth per-DOF regularization that gracefully transitions from full regularization (degenerate) to zero (well-constrained).

**Step 1: Smooth per-eigenvalue weight.**

For each eigenvalue $\lambda_i$ of the analysis Hessian:

$$
w_i = \frac{1}{1 + \left(\frac{\lambda_i / \lambda_{\max}}{\tau}\right)^{\beta}}
$$

where $\tau$ is `relative_eigenvalue_threshold` and $\beta$ is `smooth_prior_falloff`. Small eigenvalues (ratio $\ll \tau$) get $w_i \approx 1$; large eigenvalues (ratio $\gg \tau$) get $w_i \approx 0$.

**Step 2: Transform to DOF-space weights.**

$$
\hat{w}_j = \sum_{i=1}^{6} w_i \cdot v_{ji}^2
$$

where $v_{ji}$ is the $j$-th component of eigenvector $i$. This maps the eigenvalue-space weights into the physical DOF space $[\text{rot}_x, \text{rot}_y, \text{rot}_z, \text{trans}_x, \text{trans}_y, \text{trans}_z]$.

**Step 3: Anchor-aware reduction** (when `anchor_aware_regularization = true`).

The anchor Hessian $\mathbf{H}_{\text{anchor}}$ is separately eigendecomposed. For each DOF $j$, the anchor contribution is:

$$
c_j = \sum_{i} \min\!\left(\frac{\mu_i \cdot \lambda_{\text{anchor}}^{\text{eff}}}{\lambda_{\max}},\; 1\right) u_{ji}^2
$$

where $\mu_i$ are anchor eigenvalues and $u_{ji}$ are anchor eigenvector components. The regularization weight is then reduced:

$$
\hat{w}_j \leftarrow \hat{w}_j \cdot (1 - \min(c_j, 1))
$$

This means: if anchors already resolve a degenerate direction, the soft prior backs off.

**Step 4: Apply regularization.**

$$
W_{jj} = \alpha_{\max} \cdot s \cdot \hat{w}_j
$$

where $\alpha_{\max}$ is `smooth_prior_max_strength` and $s$ is a reference scale controlled by `regularization_scale_mode`:

| mode | scale $s$ | description |
|------|-----------|-------------|
| 0 | $\lambda_{\max}$ | largest eigenvalue (original-like, can be large) |
| 1 | $\text{tr}(\mathbf{H}_{\text{geom}}) / 6$ | geometry Hessian trace mean (recommended, stable) |
| 2 | $\text{mean}(\text{diag}(\mathbf{H}))$ | combined Hessian diagonal mean |

So the effective objective solved in one iteration becomes

$$
E(T,\boldsymbol{\delta}) = E_{\text{gicp}}(T) + E_{\text{anchor}}(T) + E_{\text{soft}}(\boldsymbol{\delta}).
$$

where $E_{\text{soft}}$ is either the binary suppression prior or the smooth sigmoid-based prior depending on configuration.

### Hard DOF lock

Hard lock is not added as a penalty term. Instead, the corresponding DOFs are removed from the linear solve and the accepted pose is projected back so that locked global pose components stay unchanged. In other words, hard lock is enforced as an equality constraint on the selected subset of

$$
[\mathrm{rot}_x,\mathrm{rot}_y,\mathrm{rot}_z,\mathrm{trans}_x,\mathrm{trans}_y,\mathrm{trans}_z].
$$

### Dense RGB-assisted correspondence selection

Dense RGB does not add a new residual term to the least-squares objective. Instead, on CPU `FastGICP` / `FastVGICP` it changes how correspondences are selected before the geometric objective is built.

For `FastGICP`, the code first queries the top-$K$ geometric candidates and re-ranks them by

$$
s(i,j)=d_{\text{geo}}(i,j)+\beta \frac{\left\lVert \mathbf{c}_i^s-\mathbf{c}_j^t \right\rVert_2^2}{\sigma_c^2}
$$

where $d_{\text{geo}}(i,j)$ is the squared geometric nearest-neighbor distance, $\mathbf{c}_i^s,\mathbf{c}_j^t$ are RGB vectors, $\beta$ is `color_weight`, and $\sigma_c$ is `color_sigma`.

For `FastVGICP`, the same idea is used to re-rank candidate target voxels, replacing $\mathbf{c}_j^t$ with the voxel mean color.

So RGB affects the correspondence set, while the optimized residual remains geometric GICP plus optional sparse anchors and soft suppression.

### Dynamic outlier rejection (GNC + IRLS)

When `set_dynamic_rejection_config({"enable": True, ...})` is on, each geometric correspondence (and optionally each sparse anchor) receives a per-iteration runtime weight $w_i \in [0,1]$ derived from a graduated non-convex robust kernel. The geometric residual becomes

$$
E_{\text{gicp}}^{\text{robust}}(T)=\sum_i w_i(\mu_k)\,\mathbf{e}_i(T)^\top \mathbf{\Lambda}_i(T)\mathbf{e}_i(T),
$$

where the weight under the **Geman-McClure** kernel with shape $\mu_k$ is

$$
w_i(\mu_k)=\left(\frac{\mu_k}{\mu_k + r_i^2}\right)^2,\qquad r_i^2=\mathbf{e}_i^\top \mathbf{\Lambda}_i\mathbf{e}_i.
$$

The Cauchy kernel $w_i = 1/(1+r_i^2/\mu_k)$ is also available.

$\mu_k$ follows a GNC schedule: at the first post-warmup iteration it is initialised from the MAD of the current residuals

$$
\mu_0 = \alpha_{\text{init}}\,(s\,\mathrm{MAD}(\{r_i^2\}))^{2},
$$

then shrunk every `gnc_steps_per_mu` iterations toward a floor:

$$
\mu_{k+1}=\max\!\left(\mu_k\,\rho,\ \mu_{\text{floor}}\right),
\quad
\mu_{\text{floor}}=\alpha_{\text{floor}}\,(s\,\mathrm{MAD}(\{r_i^2\}))^{2}.
$$

Here $s$ is `mad_scale` (default 1.4826), $\alpha_{\text{init}}$/$\alpha_{\text{floor}}$ are `gnc_mu_init_scale` / `gnc_mu_floor_scale`, and $\rho$ is `gnc_mu_decay`. The first `warmup_iterations` LM steps run with $w_i \equiv 1$ so a poor initial guess can settle before any rejection kicks in.

A safety floor `min_inlier_ratio` raises the smallest weights to $0.5$ when needed to ensure at least that fraction of correspondences continues to drive the optimization (prevents pathological scenes from collapsing to an under-determined system).

If `anchor_rejection_enabled` is on, the same $\mu_k$ is also applied to the sparse anchor residuals, so anchors that land on dynamic objects get downweighted consistently with geometric outliers.

**Where it is supported.** `FastGICP` (CPU, including the 2DGS surfel covariance variant), `FastGICPCuda` (GPU), and the sparse anchor path. `FastVGICP`, `FastVGICPCuda`, and `NDTCuda` throw `std::runtime_error` at config time if `enable=true` is requested, because the voxel-based correspondence model is not yet wired for IRLS.

### Multi-restart initial guess

`set_multi_restart_initial_guesses([T0, T1, ...])` lets you hand in $K$ candidate initial poses (e.g. last-frame pose + a few small SE(3) perturbations, or several IMU hypotheses). The optimizer runs `align()` from each, then keeps the result with the lowest final cost. This is the cheapest fix for "constant-velocity prior dropped into the wrong basin" failure cases that often coincide with dynamic scenes.

### Iteration method

At iteration $k$, the solver computes the linearized normal equations

$$
\mathbf{H}_k \boldsymbol{\delta}_k = -\mathbf{b}_k
$$

from the current correspondences and optional sparse anchors.

If observability-aware soft suppression is enabled, the system becomes

$$
(\mathbf{H}_k + \mathbf{W}_k)\boldsymbol{\delta}_k = -\mathbf{b}_k.
$$

In binary mode, $\mathbf{W}_k$ is nonzero only on DOFs classified as degenerate. In smooth mode, $\mathbf{W}_k$ is smooth across all DOFs based on the eigenvalue ratio sigmoid.

If Levenberg-Marquardt is used, damping is further added on the unlocked DOFs:

$$
(\mathbf{H}_k + \mathbf{W}_k + \lambda_k \mathbf{I})\boldsymbol{\delta}_k = -\mathbf{b}_k.
$$

If hard lock is enabled, the locked rows/columns are excluded from the solve and those pose components are restored after the update.

The pose update is left-multiplicative on $SE(3)$:

$$
T_{k+1}=\exp(\hat{\boldsymbol{\delta}}_k)T_k.
$$

Then, if hard lock is active, the accepted pose is projected so the locked pose components remain equal to the previous pose. Convergence is checked from the incremental rotation and translation magnitudes, same as before.

New in this branch:

- Observability diagnostics on the 6x6 Hessian.
- Automatic soft suppression for ambiguous DOFs (binary mode).
- Smooth sigmoid-based observability prior with anchor-aware reduction and geometry-only analysis.
- Explicit hard-lock for any subset of $[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]$.
- Sparse visual anchor constraints from externally matched 3D-3D keypoints.
- Optional RGB-assisted correspondence selection on CPU `FastGICP` / `FastVGICP`.
- **Dynamic-scene outlier rejection** (GNC + IRLS with Geman-McClure / Cauchy kernel,
  MAD-driven mu schedule, warmup, anchor co-rejection, min-inlier-ratio floor).
- **Multi-restart initial guess** + optional basin-suspect post-check.
- **`FastGICPCuda`** new CUDA class mirroring classic GICP / 2DGS surfel covariance /
  sparse 3D anchors / dynamic rejection on the GPU. Independent of the legacy
  nvbio-based VGICP-CUDA library (which is now gated by an opt-in flag).
- Optional **cuVS KNN backend** (`set_knn_backend("cuvs")`, requires
  `-DUSE_CUVS=ON`).

All of the new functionality is disabled by default. When disabled, the new code path keeps the original behavior of this branch.

## Feature support matrix

| Feature | FastGICP | FastVGICP | FastGICPCuda | FastVGICP_CUDA | NDT_CUDA |
| --- | --- | --- | --- | --- | --- |
| Hessian observability diagnostics | yes | yes | yes | yes | yes |
| Auto soft suppression / hard DOF lock | yes | yes | yes | yes | yes |
| Smooth observability prior | yes | yes | yes | yes | yes |
| Sparse 3D visual anchors | yes | yes | yes | yes | yes |
| Dense RGB-assisted matching | yes | yes | no | no | no |
| 2DGS surfel covariance (from quat + 2D scales) | yes | inherited | yes | no | no |
| **Dynamic outlier rejection (GNC/IRLS)** | **yes** | **throws** | **yes** | **throws** | **throws** |
| Multi-restart initial guess | yes | yes | yes | yes | yes |
| KNN backend selector (brute_force / cuvs) | n/a | n/a | yes | n/a | n/a |

`FastVGICP*`/`NDT_CUDA` throw `std::runtime_error` from `setDynamicRejectionConfig`
when `enable=true`, because voxel correspondences are not wired for per-point
IRLS weights.

Install for python
```shell
cd catkin_workspace
catkin_make -DCMAKE_BUILD_TYPE=Release
cd src/fast_gicp
python3 setup.py install --user
```

python usage (see src/fast_gicp/python):

```python
import pygicp

gicp = pygicp.FastGICP()

gicp.set_input_target(target)
gicp.set_input_source(source)

# set covariance from quaternion and scale by following normalized_elipse
nparray_of_quaternions = nparray_of_quaternions_Nx4.flatten()
nparray_of_scales = nparray_of_scales_NX3.flatten()
gicp.set_target_covariance_fromqs(nparray_of_quaternions, nparray_of_scales) => 0.002180 sec
gicp.set_source_covariance_fromqs(nparray_of_quaternions, nparray_of_scales)

# compute covariance by following normalized_elipse
calculate_target_covariance() # compute covariance from given input target pointcloud
calculate_source_covariance() # compute covariance from given input source pointcloud

# after gicp.align()
correspondences, sq_distances = gicp.get_source_correspondence()
covariances = get_target_covariances()
covariances = get_source_covariances()
nparray_of_quaternions = get_target_rotationsq() => 0.00002277 sec
nparray_of_quaternions = get_source_rotationsq() 
nparray_of_scales = get_target_scales()          => 0.00002739 sec
nparray_of_scales = get_source_scales()
nparray_of_quaternions_Nx4 = np.reshape(nparray_of_quaternions, (-1,4))
nparray_of_scales_NX3 = np.reshape(nparray_of_scales, (-1,3))

# observability-aware optimization
gicp.set_observability_check(True)
gicp.set_enable_observability_diagnostics(True)
gicp.set_observability_eigen_thresholds(0.1, 1e-6)
gicp.set_auto_soft_prior_strength(0.02)

# prefer suppressing trans_z when the Hessian indicates ambiguity
gicp.set_preferred_ambiguous_mask(np.array([0, 0, 0, 0, 0, 1], dtype=np.int32))

# hard-lock trans_z
gicp.set_hard_lock_mask(np.array([0, 0, 0, 0, 0, 1], dtype=np.int32))

# sparse visual anchors from an external SIFT / ORB / LightGlue pipeline
# source_points_3d and target_points_3d must be Nx3 arrays in the same metric frame as the point clouds
gicp.set_sparse_anchor_correspondences(source_points_3d, target_points_3d, weights=None, sigmas=None)
gicp.set_sparse_anchor_balance_mode("BY_HESSIAN_TRACE")
gicp.set_sparse_anchor_objective_weight(1.0)

# optional RGB-assisted correspondence selection on CPU FastGICP / FastVGICP
gicp.set_source_colors(source_rgb)  # Nx3, float or uint8-like values
gicp.set_target_colors(target_rgb)
gicp.set_color_matching(True, candidate_count=5, color_weight=0.5, color_sigma=16.0)

# inspect the final Hessian diagnostics
diag = gicp.get_observability_diagnostics()
print(diag["estimated_rank"], diag["ambiguity_scores"])
print(diag["geometry_eigenvalues"], diag["smooth_regularization_weights"])

# smooth observability prior (alternative to binary auto_soft_prior)
gicp.set_observability_check(True)
gicp.set_use_smooth_prior(True)
gicp.set_smooth_prior_falloff(2.0)                # sigmoid steepness
gicp.set_smooth_prior_max_strength(0.1)           # max regularization relative to scale
gicp.set_analyze_geometry_separately(True)         # eigendecompose geometry Hessian alone
gicp.set_anchor_aware_regularization(True)         # reduce reg where anchors help
gicp.set_regularization_scale_mode(1)              # 0=lambda_max, 1=geom_trace, 2=diag_mean

# optional per-align quality report for SLAM front-end gating
gicp.set_alignment_quality_config({
    "enable_suggested_gating": True,
    "min_matched_count": 50,
    "min_matched_ratio": 0.1,
    "max_p95_sq_distance": 0.05 ** 2,
    "min_rank": 4,
    "max_dof_ambiguity": np.array([1.0, 1.0, 0.8, 1.0, 1.0, 0.5]),
})
gicp.align()
report = gicp.get_alignment_quality_report()
print(report["suggested_accept"], report["rejection_reasons"])
if report["used_sparse_anchors"]:
    print(
        report["anchor_balance_mode"],
        report["anchor_objective_weight"],
        report["anchor_auto_balance_factor"],
        report["anchor_effective_scale"],
    )
    print(
        report["geometry_hessian_trace"],
        report["anchor_hessian_trace"],
        report["anchor_balance_fallback_used"],
    )

```

### Dynamic outlier rejection + multi-restart (CPU and CUDA)

```python
import pygicp
import numpy as np

# Works on both CPU FastGICP and FastGICPCuda (and the 2DGS surfel covariance
# variant of FastGICP). FastVGICP* and NDT_CUDA throw if enable=True.
reg = pygicp.FastGICP()                   # or pygicp.FastGICPCuda() if built
reg.set_input_target(target_xyz_nx3)
reg.set_input_source(source_xyz_nx3)
reg.set_max_correspondence_distance(2.0)

reg.set_dynamic_rejection_config({
    "enable": True,
    "kernel": "GEMAN_MCCLURE",        # or "CAUCHY", or "NONE"
    "warmup_iterations": 3,            # LM iters before rejection kicks in
    "mad_scale": 1.4826,               # MAD -> sigma conversion
    "gnc_mu_init_scale": 9.0,          # initial mu = (mad*scale)^2 * this
    "gnc_mu_floor_scale": 1.0,         # floor mu = (mad*scale)^2 * this
    "gnc_mu_decay": 0.5,               # mu *= decay every gnc_steps_per_mu iters
    "gnc_steps_per_mu": 2,
    "min_inlier_ratio": 0.2,           # safety floor: at least 20% keep weight >=0.5
    "anchor_rejection_enabled": True,  # apply same kernel to sparse anchors
    "basin_verify_enabled": False,     # add basin_suspect to rejection_reasons
    "basin_suspect_cost_ratio": 0.5,
})

# Optional multi-restart from K initial-guess candidates; the best (lowest final
# cost) is kept and reported through the usual getters.
reg.set_multi_restart_initial_guesses([
    T_prev_velocity.astype(np.float32),
    np.eye(4, dtype=np.float32),
    T_prev_velocity_perturbed.astype(np.float32),
])

T = reg.align(T_prev_velocity.astype(np.float32))

# Diagnostics
diag = reg.get_dynamic_rejection_diagnostics()
# diag keys: enabled, active, gnc_iterations, final_mu, inlier_sigma,
#            total_correspondences, inlier_count, anchor_total,
#            anchor_inlier_count, mean_inlier_residual, mean_outlier_residual
weights   = np.asarray(reg.get_dynamic_correspondence_weights())   # length N source, [0,1]
residuals = np.asarray(reg.get_dynamic_correspondence_residuals()) # Mahalanobis r^2
anchor_w  = np.asarray(reg.get_dynamic_anchor_weights())           # length N anchors

# Restore default behavior
reg.set_dynamic_rejection_enabled(False)
reg.clear_multi_restart_initial_guesses()
```

### FastGICPCuda (GPU classic / 2DGS / anchors)

```python
import pygicp                # built with -DBUILD_VGICP_CUDA=ON

g = pygicp.FastGICPCuda()
g.set_correspondence_randomness(20)
g.set_max_correspondence_distance(2.0)
g.set_input_target(target_xyz_nx3)   # uploaded to device once per call
g.set_input_source(source_xyz_nx3)

# Optional KNN backend; "cuvs" only available when built with -DUSE_CUVS=ON.
g.set_knn_backend("brute_force")     # default; recommended for typical N
# g.set_knn_backend("cuvs")          # uses cuVS brute_force (cuBLAS GEMM)

# Optional 2DGS surfel covariances (same semantics as the CPU path):
g.set_source_covariances_from_2dgs(rotations_xyzw_nx4.astype(np.float32).ravel(),
                                    scales_2d_nx2.astype(np.float32).ravel(),
                                    "physical", 0.05, 1e-3)
g.set_target_covariances_from_2dgs(target_rotations_xyzw_nx4.astype(np.float32).ravel(),
                                    target_scales_2d_nx2.astype(np.float32).ravel(),
                                    "physical", 0.05, 1e-3)

# All LsqRegistration features apply: dynamic rejection, multi-restart,
# sparse anchors, alignment quality config, observability check.
g.set_dynamic_rejection_config({"enable": True, "kernel": "GEMAN_MCCLURE",
                                "warmup_iterations": 3, "min_inlier_ratio": 0.2})
g.set_sparse_anchor_correspondences(src_anchors_nx3, tgt_anchors_nx3, weights, sigmas)

T = g.align(np.eye(4, dtype=np.float32))
print("source/target on device:", g.get_source_size(), g.get_target_size())
print("knn backend:", g.get_knn_backend())
```

## C++ API additions

Useful new entry points on `fast_gicp::LsqRegistration`:

- `setObservabilityCheck(bool)`
- `setObservabilityEigenThresholds(double relative, double absolute)`
- `setAutoSoftPriorStrength(double)`
- `setObservabilityConfig(config)` / `getObservabilityConfig()`
- `setHardLockMask(std::array<int, 6>)`
- `setPreferredAmbiguousMask(std::array<int, 6>)`
- `setSparseAnchorCorrespondences(source_points, target_points, weights, sigmas)`
- `setSparseAnchorConfig(...)`
- `getSparseAnchorConfig()`
- `setSparseAnchorObjectiveWeight(double)`
- `setSparseAnchorBalanceMode(...)`
- `getObservabilityDiagnostics()`
- `getFinalRegularizedHessian()`
- `setAlignmentQualityConfig(...)`
- `getAlignmentQualityConfig()`
- `getAlignmentQualityReport()`
- `setDynamicRejectionConfig(const DynamicRejectionConfig&)`
- `setDynamicRejectionEnabled(bool)`
- `getDynamicRejectionConfig()` / `getDynamicRejectionDiagnostics()`
- `getDynamicCorrespondenceWeights()` / `getDynamicCorrespondenceResiduals()`
- `getDynamicAnchorWeights()` / `getDynamicAnchorResiduals()`
- `setMultiRestartInitialGuesses(const std::vector<Eigen::Matrix4f>&)`
- `clearMultiRestartInitialGuesses()`
- `supports_dynamic_rejection()` (virtual): override returns `false` on
  `FastVGICP`/`FastVGICPCuda`/`NDTCuda`, `true` on `FastGICP`/`FastGICPCuda`.
  `setDynamicRejectionConfig({enable: true})` throws on unsupported classes.

The `ObservabilityConfig` struct includes the smooth prior fields:
`use_smooth_prior`, `smooth_prior_falloff`, `smooth_prior_max_strength`,
`analyze_geometry_separately`, `anchor_aware_regularization`, `regularization_scale_mode`.

`DynamicRejectionConfig` fields (declared in `gicp_settings.hpp`):
`enable`, `kernel` (`DynamicRejectionKernel::NONE|GEMAN_MCCLURE|CAUCHY`),
`warmup_iterations`, `mad_scale`, `gnc_mu_init_scale`, `gnc_mu_floor_scale`,
`gnc_mu_decay`, `gnc_steps_per_mu`, `min_inlier_ratio`, `anchor_rejection_enabled`,
`basin_verify_enabled`, `basin_suspect_cost_ratio`.

Useful new entry points on CPU `fast_gicp::FastGICP` / `fast_gicp::FastVGICP`:

- `setSourceColors(...)`
- `setTargetColors(...)`
- `setColorMatchingConfig(...)`
- `setSourceCovariances2DGS(rotationsq_xyzw, scales_2d, mode, normal_sigma_ratio, normal_sigma_min)`
- `setTargetCovariances2DGS(...)`

New CUDA class `fast_gicp::FastGICPCuda<PointSource, PointTarget>` (header
`fast_gicp/gicp/fast_gicp_cuda.hpp`, opt-in `-DBUILD_VGICP_CUDA=ON`):

- `setCorrespondenceRandomness(int)` / `setRegularizationMethod(RegularizationMethod)`
- `setKnnBackend(const std::string&)` (`"brute_force"` always, `"cuvs"` if
  `-DUSE_CUVS=ON`) / `getKnnBackend()`
- `setInputSource` / `setInputTarget` (host PCL clouds; copied to device)
- `setSourceCovariances2DGS(...)` / `setTargetCovariances2DGS(...)`
- `getSourceSize()` / `getTargetSize()`
- All `LsqRegistration` features (dynamic rejection, multi-restart, sparse
  anchors, observability, quality report) work unchanged.

The PImpl handle `cuda::FastGICPCudaCore` (declared in
`fast_gicp/cuda/fast_gicp_cuda_core.cuh`) keeps all `thrust::device_vector`s
out of consumer translation units; only `.cu` files need to include it.

## New parameter guide

### Observability and soft suppression

- `set_observability_check(bool)` / `setObservabilityCheck(bool)`
  - Meaning: turn on Hessian eigen-analysis and ambiguity scoring.
  - Recommended: `True` for planar, corridor, wall-only, or repeated-texture scenes.
  - Keep `False` if you want the exact old behavior.

- `set_enable_observability_diagnostics(bool)` / `setEnableObservabilityDiagnostics(bool)`
  - Meaning: store the raw Hessian, regularized Hessian, eigenvalues, ambiguity scores, selected suppressed DOFs, and estimated rank.
  - Recommended: `True` during debugging and parameter tuning, optional in production.

- `set_observability_eigen_thresholds(relative, absolute)` / `setObservabilityEigenThresholds(relative, absolute)`
  - Meaning: define which Hessian eigenvalues are treated as weakly observable.
  - `relative`: threshold relative to the largest eigenvalue.
  - `absolute`: floor for very small Hessians.
  - Recommended start: `relative=0.01 ~ 0.1`, `absolute=1e-9 ~ 1e-6`.
  - Tuning:
    - Increase `relative` if ambiguous DOFs are not being detected.
    - Decrease `relative` if too many DOFs are classified as ambiguous.

### Smooth observability prior

When the binary soft suppression is too coarse (either does nothing or constrains all frames), the smooth prior provides a gradual alternative.

- `set_use_smooth_prior(bool)`
  - Meaning: enable the smooth sigmoid-based prior instead of the binary threshold.
  - Requires `observability_check = True`.
  - Recommended: `True` when the binary mode either fires on no frames or constrains all frames.

- `set_smooth_prior_falloff(beta)` / default: `2.0`
  - Meaning: exponent $\beta$ in the sigmoid $w = 1/(1 + (r/\tau)^\beta)$.
  - Higher values make the transition sharper (closer to binary); lower values make it softer.
  - Recommended range: `1.0 ~ 4.0`.

- `set_smooth_prior_max_strength(alpha_max)` / default: `0.1`
  - Meaning: maximum regularization weight relative to the reference scale.
  - Recommended start: `0.05 ~ 0.2`.
  - Increase if drift along degenerate directions remains visible.
  - Decrease if optimization becomes too sluggish.

- `set_analyze_geometry_separately(bool)` / default: `false`
  - Meaning: eigendecompose the geometry Hessian alone (without anchors) to detect degeneracy.
  - Recommended: `True` when using sparse anchors, so the check detects what GICP alone cannot resolve.
  - When `false`, the combined Hessian is analyzed (original behavior).

- `set_anchor_aware_regularization(bool)` / default: `false`
  - Meaning: reduce the smooth regularization in directions where the sparse anchor term already provides information.
  - Recommended: `True` when `analyze_geometry_separately = True` and sparse anchors are enabled.
  - The reduction is proportional to the anchor Hessian contribution in each direction.

- `set_regularization_scale_mode(mode)` / default: `0`
  - Meaning: choose the reference scale $s$ for the regularization magnitude $W_{jj} = \alpha_{\max} \cdot s \cdot \hat{w}_j$.
  - `0`: $s = \lambda_{\max}$ (original-like, can be very large for well-constrained directions).
  - `1`: $s = \text{tr}(\mathbf{H}_{\text{geom}}) / 6$ (geometry trace mean, recommended — stable).
  - `2`: $s = \text{mean}(\text{diag}(\mathbf{H}))$ (combined Hessian diagonal mean).
  - Recommended: `1` for most cases.

- `set_auto_soft_prior_strength(alpha)` / `setAutoSoftPriorStrength(alpha)`
  - Meaning: scale of the automatic diagonal prior added to ambiguous DOFs.
  - Recommended start: `0.01 ~ 0.05`.
  - Tuning:
    - Increase it if drift along ambiguous DOFs is still visible.
    - Decrease it if optimization becomes too conservative or lags behind true motion.

- `set_preferred_ambiguous_mask(mask)` / `setPreferredAmbiguousMask(mask)`
  - Meaning: bias the auto-suppression policy toward the listed DOFs when ambiguity exists.
  - Mask order:
    - `[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]`
  - Recommended use:
    - Wall / fridge / cabinet scenes: prefer `trans_z`.
    - Ground-only scenes: often prefer `rot_x`, `rot_y`, `trans_z`.
    - Long corridors: often prefer lateral drift or yaw depending on geometry.

### Hard lock

- `set_hard_lock_mask(mask)` / `setHardLockMask(mask)`
  - Meaning: remove selected DOFs from the linear solve and project the accepted pose so those pose components stay fixed.
  - Mask order:
    - `[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]`
  - Recommended use:
    - Use only when you know a DOF should be fixed by design, or when auto soft suppression is still insufficient.
  - Tuning:
    - Start with preferred ambiguous mask + soft suppression first.
    - Escalate to hard lock only for persistent failure modes.

### Sparse 3D anchors

- `set_sparse_anchor_correspondences(source_points, target_points, weights=None, sigmas=None)` / `setSparseAnchorCorrespondences(...)`
  - Meaning: add externally matched 3D-3D constraints from SIFT / ORB / LightGlue or other visual frontends.
  - Inputs:
    - `source_points`, `target_points`: `Nx3`, same metric coordinate convention as the point clouds.
    - `weights`: optional confidence per match.
    - `sigmas`: optional uncertainty scale per match.
  - Recommended use:
    - Use only high-confidence matches after 2D matching, depth backprojection, and outlier filtering.
  - Tuning:
    - Increase `weights` for highly trusted anchors.
    - Increase `sigmas` or reduce `weights` if anchors over-pull the solution.
    - Prefer a small number of reliable anchors over many noisy ones.

- `set_sparse_anchor_balance_mode(mode)` / `setSparseAnchorBalanceMode(...)`
  - Available modes:
    - `NONE`: keep the historical behavior with no extra balancing
    - `BY_COUNT`: scale anchors by geometric residual count versus anchor count
    - `BY_HESSIAN_TRACE`: scale anchors by unlocked geometry/anchor Hessian trace
  - Recommended default: `BY_HESSIAN_TRACE`

- `set_sparse_anchor_objective_weight(weight)` / `setSparseAnchorObjectiveWeight(...)`
  - Meaning: explicit global multiplier for the sparse anchor objective.
  - Recommended start: `1.0`
  - Increase it if anchors are still too weak after balancing.
  - Decrease it if anchors dominate too aggressively.

- `set_sparse_anchor_config(dict)` / `setSparseAnchorConfig(...)`
  - Fields:
    - `objective_weight`
    - `balance_mode`
    - `auto_balance_min`
    - `auto_balance_max`

- Sparse-anchor balance diagnostics are included in `get_alignment_quality_report()` whenever an align call has run.
  - Useful fields:
    - `anchor_balance_mode`: which balancing rule was used
    - `anchor_objective_weight`: user-provided multiplier
    - `anchor_auto_balance_factor`: automatically computed balance factor
    - `anchor_effective_scale`: final anchor scale applied inside the objective, equal to `anchor_objective_weight * anchor_auto_balance_factor`
    - `geometry_hessian_trace`: unlocked trace of the geometry Hessian used by `BY_HESSIAN_TRACE`
    - `anchor_hessian_trace`: unlocked trace of the anchor Hessian used by `BY_HESSIAN_TRACE`
    - `anchor_balance_fallback_used`: `True` when auto balancing had to fall back to a neutral factor because the statistics were not usable

### Dense RGB correspondence

- `set_source_colors(colors)` / `setSourceColors(...)`
- `set_target_colors(colors)` / `setTargetColors(...)`
  - Meaning: provide per-point RGB-like vectors for source and target.
  - Inputs:
    - `Nx3`, aligned with the point order of the source and target clouds.

- `set_color_matching(enabled, candidate_count, color_weight, color_sigma)` / `setColorMatchingConfig(...)`
  - Meaning: re-rank geometric correspondence candidates using RGB consistency.
  - Parameters:
    - `enabled`: switch RGB-assisted re-ranking on or off.
    - `candidate_count`: number of geometric nearest-neighbor candidates considered before RGB re-ranking.
    - `color_weight`: relative importance of color versus geometry.
    - `color_sigma`: normalization scale for RGB distance.
  - Recommended start:
    - `candidate_count=3 ~ 8`
    - `color_weight=0.1 ~ 1.0`
    - `color_sigma=8 ~ 32` for `0~255` RGB
    - `color_sigma=0.05 ~ 0.2` for normalized `0~1` RGB
  - Tuning:
    - Increase `candidate_count` if geometry alone often picks the wrong local match.
    - Increase `color_weight` if color should dominate more strongly.
    - Decrease `color_weight` if lighting or auto exposure makes color unreliable.

### Alignment quality report

- `set_alignment_quality_config(dict)` / `setAlignmentQualityConfig(...)`
  - Meaning: configure advisory gating thresholds without changing `align()` behavior.
  - The report is still generated even when advisory gating is disabled.

- `get_alignment_quality_report()` / `getAlignmentQualityReport()`
  - Meaning: fetch the most recent per-align report.
  - Core fields:
    - convergence: `converged`, `num_iterations`, `fitness_score`, `final_cost`
    - match support: `correspondence_count`, `matched_count`, `matched_ratio`
    - residual distribution: `mean_sq_distance`, `median_sq_distance`, `p90_sq_distance`, `p95_sq_distance`
    - observability: `rank`, `condition_number`, `ambiguity_scores`, `max_ambiguity`, `geometry_rank`, `geometry_condition_number`, `smooth_regularization_weights`
    - anchor diagnostics: `anchor_count`, `anchor_mean_residual`, `anchor_p95_residual`, `anchor_balance_mode`, `anchor_objective_weight`, `anchor_auto_balance_factor`, `anchor_effective_scale`, `geometry_hessian_trace`, `anchor_hessian_trace`, `anchor_balance_fallback_used`
    - advisory result: `suggested_accept`, `rejection_reasons`
  - If sparse anchors are enabled, the balance-related fields let you inspect all three layers of scaling:
    - `anchor_objective_weight`: manual top-level weight
    - `anchor_auto_balance_factor`: auto-computed factor from the selected balance mode
    - `anchor_effective_scale`: the actual multiplier applied to the sparse-anchor term
  - DOF order is always `[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]`.
  - Recommended SLAM use:
    - do not trust `converged` alone
    - reject or downweight alignments with high tail residuals or high ambiguity on critical DOFs such as `trans_z` or yaw
    - Set `color_sigma` to match your RGB numeric range; too small makes color overly sharp, too large makes it ineffective.

### Dynamic outlier rejection

- `set_dynamic_rejection_config(dict)` / `setDynamicRejectionConfig(cfg)`
  - Meaning: enable GNC-IRLS robust kernel on geometric correspondences (and
    optionally sparse anchors). Keys: see `DynamicRejectionConfig` above.
  - Sensible starting point: `{"enable": True, "kernel": "GEMAN_MCCLURE",
    "warmup_iterations": 3, "min_inlier_ratio": 0.2}` for indoor SLAM with
    typical noise ~ a few mm. Increase `min_inlier_ratio` (e.g. 0.4) if you
    only expect a small dynamic fraction and want to protect against the
    rejection over-trimming the optimizer.
- `set_dynamic_rejection_enabled(bool)`: toggle without touching other fields.
- Diagnostics: `get_dynamic_rejection_diagnostics()` reports `final_mu`,
  `inlier_sigma`, per-side inlier counts, and mean inlier/outlier residual.

### Multi-restart initial guess

- `set_multi_restart_initial_guesses([T0, T1, ...])`: provide $K$ candidate
  4x4 float poses. The optimizer runs `align()` from each and keeps the result
  with the lowest final cost. Returned pose / Hessian / quality report all
  reflect the chosen restart.
- `clear_multi_restart_initial_guesses()`: revert to single-guess mode.
- Useful when constant-velocity priors land in the wrong basin (e.g. on
  scenes with repetitive geometry like corridors), often combined with
  `basin_verify_enabled`.

### Practical tuning order

1. Keep all new features off and confirm the baseline behavior.
2. Enable observability check and diagnostics.
3. Try the **smooth prior** first: `use_smooth_prior=True`, `analyze_geometry_separately=True`, `regularization_scale_mode=1`.
4. If using sparse anchors, enable `anchor_aware_regularization=True`.
5. If smooth prior is insufficient, fall back to binary: set `use_smooth_prior=False`, add a small `auto_soft_prior_strength`, and set `preferred_ambiguous_mask`.
6. If drift remains, add sparse anchors.
7. If the scene has useful texture/color but weak geometry, enable dense RGB correspondence.
8. Only use hard lock for persistent, well-understood failure modes.

Recommended policy for simple vertical-plane scenes:

- Enable observability check.
- Enable smooth prior with `analyze_geometry_separately=True` and `regularization_scale_mode=1`.
- If using sparse anchors, enable `anchor_aware_regularization=True`.
- If smooth prior alone is insufficient, additionally mark `trans_z` in `preferred_ambiguous_mask`.
- If your application requires strict suppression, hard-lock `trans_z`.
- If you have reliable visual matches, add sparse anchors before alignment.

## Build and test

### CMake options at a glance

| Option | Default | Effect |
| --- | --- | --- |
| `BUILD_PYTHON_BINDINGS` | OFF | Build the `pygicp` shared module. |
| `BUILD_apps` | ON | Build the legacy demo apps (turn off in headless envs). |
| `BUILD_test` | OFF | Build `gicp_selfcheck` and the new `dynamic_rejection_test` C++ self-tests. |
| `BUILD_benchmarks` | OFF | Build the `gicp_benchmark` driver. |
| `BUILD_VGICP_CUDA` | OFF | Build the new `FastGICPCuda` class + kernels (`fast_gicp_cuda_kernels` SHARED). Required for `pygicp.FastGICPCuda`. |
| `FAST_GICP_BUILD_LEGACY_VGICP_CUDA` | OFF | Also build the legacy nvbio-based `fast_vgicp_cuda` library (`FastVGICPCuda` / `NDTCuda`). It pulls in nvbio + an older thrust which currently has a `thrust::pair` ambiguity under CUDA 12.x. Leave OFF unless you specifically need the voxel CUDA path. |
| `USE_CUVS` | OFF | Compile in the cuVS brute-force KNN backend. Requires `libcuvs_c.so` + `libcuvs.so` + `dlpack/dlpack.h` (e.g. `mamba install -c rapidsai -c conda-forge libcuvs cuvs dlpack cuda-version=12.8`). |
| `FAST_GICP_CUDA_ARCH` | `120` | CUDA architectures (semicolon-separated, no `sm_` prefix). Default targets Blackwell sm_120; pass `"75;86;89;120"` to span several GPUs. |

### CPU-only build (recommended for the dynamic-rejection feature alone)

```bash
source /nvme1/jiaheng/miniconda3/etc/profile.d/conda.sh
conda activate nsl2

mkdir -p build_dyn
cmake -S . -B build_dyn \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_apps=OFF \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_test=ON \
  -DPCL_DIR=$CONDA_PREFIX/share/pcl-1.14 \
  -DBOOST_ROOT=$CONDA_PREFIX \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build_dyn -j$(nproc)

# Tests
./build_dyn/dynamic_rejection_test                              # C++ self-tests
PYTHONPATH=build_dyn pytest python_tester/dynamic -v            # 12 pytest cases
```

The C++ helper `dynamic_rejection_test` covers MAD, GNC mu schedule, kernel
monotonicity, and the inlier-ratio floor.

### CUDA build (FastGICPCuda, optional cuVS)

```bash
mkdir -p build_dyn_cuda
cmake -S . -B build_dyn_cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_apps=OFF \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_test=ON \
  -DBUILD_VGICP_CUDA=ON \
  -DUSE_CUVS=ON \
  -DFAST_GICP_CUDA_ARCH="120" \
  -DPCL_DIR=$CONDA_PREFIX/share/pcl-1.14 \
  -DBOOST_ROOT=$CONDA_PREFIX \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build_dyn_cuda -j$(nproc)

# Tests (11 CUDA cases including the 500K-scale speedup benchmark + cuVS parity)
PYTHONPATH=build_dyn_cuda pytest python_tester/dynamic_cuda -v -s
```

Drop `-DUSE_CUVS=ON` if you don't have cuVS installed; the build still
produces a working `FastGICPCuda` with the `brute_force` KNN backend.
Pass `-DFAST_GICP_BUILD_LEGACY_VGICP_CUDA=ON` only if you also want the
voxel CUDA classes (the legacy nvbio path is independent of this branch's
new work).

### Install isolation (do **not** run `setup.py install` while developing)

The conda env's `pygicp` lives at
`$CONDA_PREFIX/lib/python3.11/site-packages/` (editable wheels point its
`.so` at whichever sibling fast_gicp checkout you originally installed
from). Running `setup.py install` in this checkout would silently overwrite
that `.so` and disrupt downstream SLAM code that relies on the older API.

For development we always build into `build_dyn/` or `build_dyn_cuda/`
without installing, then load via `PYTHONPATH`:

```python
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "build_dyn"))         # or build_dyn_cuda
import pygicp
assert "build_dyn" in pygicp.__file__, "wrong pygicp; install isolation broken"
```

The two test directories (`python_tester/dynamic`, `python_tester/dynamic_cuda`)
ship `_loader.py` shims that enforce exactly this assertion before any test
runs.

### Self-check + benchmark on the legacy app harness

```bash
./build_dyn/gicp_selfcheck
./build_dyn/gicp_benchmark ./data 20
```

The benchmark prints `baseline_default`, `robust_observability`, and
`robust_observability_anchor`. `baseline_default` keeps every new feature
disabled (dynamic rejection too).

### Performance summary (RTX PRO 6000, source ~ 20K)

| target size | CPU 16-thread | CUDA brute_force | CUDA cuVS | speedup (bf / cuvs) |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 29 ms | 21 ms | 18 ms | 1.39x / 1.65x |
| 50,000 | 80 ms | 51 ms | 49 ms | 1.57x / 1.61x |
| 200,000 | 200 ms | 139 ms | 210 ms | 1.45x / 0.96x |
| 500,000 | 485 ms | 432 ms | 904 ms | 1.12x / 0.54x |

cuVS uses cuBLAS GEMM to materialise a tiled pairwise distance matrix; for
GICP's fixed small K (=20 for covariance build, K=1 for per-iter
correspondence) the tiling overhead dominates beyond ~100K targets and the
hand-rolled per-thread kernel wins. cuVS is most useful when K grows large
or when you want approximate methods (IVF/CAGRA) instead of brute force.

The S3 corridor scene (corridor + sliding cabinet) shows the dynamic
rejection's headline result on CPU:

| variant | rotation error | translation error |
| --- | ---: | ---: |
| baseline GICP | 0.044 deg | **0.586 m** (drifts along corridor) |
| GICP + dynamic rejection | 0.042 deg | **0.004 m** (~150x better) |

CUDA recovers the same scene to ~0.066 m (float-vs-double precision) — a
9x improvement over the CUDA no-rejection baseline.
## Dynamic-GICP synthetic test suites

Two pytest suites verify the new functionality on synthetic indoor scenes
(no external dataset required — geometry is generated procedurally with a
fixed seed). Each test writes per-scene PNG visualizations plus an
aggregated `results/index.html` gallery.

```bash
# CPU (12 cases, 10 indoor scenes S1-S10 + perf overhead test)
PYTHONPATH=build_dyn pytest python_tester/dynamic -v

# CUDA (11 cases, including S3 corridor parity + 500K-scale speedup sweep
# + cuVS vs brute_force parity check)
PYTHONPATH=build_dyn_cuda pytest python_tester/dynamic_cuda -v -s
```

Scene catalogue (see `python_tester/dynamic/synthetic_indoor.py`):

| ID | What it tests |
| --- | --- |
| S1 | room + one large mover (~15% dynamic) — basic rejection |
| S2 | room + 4 moving cylinders ("pedestrians") — multi-object |
| S3 | corridor + sliding cabinet — repetitive geometry + dynamic, **the killer case** for translation drift |
| S4 | symmetric "mirror" room — convergence basin study |
| S5 | planar-floor degeneracy — rejection must not over-trim |
| S6 | static room with large init perturbations — basin sweep |
| S7 | dynamic outliers + large init offset combined — worst case |
| S8 | sparse anchors with bad anchors on movers — anchor recall |
| S9 | 2DGS surfel covariance path |
| S10 | static regression — rejection must not degrade clean scenes |
| 500K speedup (CUDA only) | sweeps target = 10K / 50K / 200K / 500K, plots `speedup_summary.png` |

Per-scene artefacts (`python_tester/dynamic{,_cuda}/results/*.png`):
3D overview, GT-dynamic-mask overlay, runtime-weight overlay with TP/FP/FN
confusion, pose-error bars, Mahalanobis residual histogram, basin
heatmaps (S4/S6), and the CUDA speedup summary.

## fast_gicp_tester
- We provide simple test code on Replica/TUM dataset.
```bash
cd python_tester
```

### Download dataset
One need to download datasets for testing. To download Replica and TUM dataset, use bash files.
```bash
# Replica
bash download_replica.sh
# TUM
bash download_tum.sh
```

### Usage

```bash
python gicp_odometry2.py [dataset_path] [tum or replica] [downsample_resolution] [visualize?]
```

### Example
Test fast-gicp on replica room0, with random downsampling ratio 0.05. And visualize registered pointclouds.

```bash
python gicp_odometry2.py ./dataset/Replica/room0 replica 0.05 true
```
<p align="center">
  <img width="35%" src="https://github.com/Lab-of-AI-and-Robotics/fast_gicp/blob/main/data/replica_0.05_traj.png"/>
  <img width="40%" src="https://github.com/Lab-of-AI-and-Robotics/fast_gicp/blob/main/data/replica_0.05_pointcloud.gif"/>
</p>




Test fast-gicp on TUM_fr3_office, without random downsampling and visualizing pointclouds.
```bash
python gicp_odometry2.py ./dataset/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household tum false false
```

Test fast-gicp on TUM_fr3_office, with random downsampling ratio 0.05, and visualizing pointclouds. 
```bash
python using_previous_30.py dataset/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household tum 0.05 true
```

<p align="center">
  <img width="40%" src="https://github.com/Lab-of-AI-and-Robotics/fast_gicp/blob/main/data/tum_30_elipse.png"/>
  <img width="40%" src="https://github.com/Lab-of-AI-and-Robotics/fast_gicp/blob/main/data/tum_30_elipse.gif"/>
</p>
