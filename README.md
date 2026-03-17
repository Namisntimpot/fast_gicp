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

So the effective objective solved in one iteration becomes

$$
E(T,\boldsymbol{\delta}) = E_{\text{gicp}}(T) + E_{\text{anchor}}(T) + E_{\text{soft}}(\boldsymbol{\delta}).
$$

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
- Automatic soft suppression for ambiguous DOFs.
- Explicit hard-lock for any subset of $[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]$.
- Sparse visual anchor constraints from externally matched 3D-3D keypoints.
- Optional RGB-assisted correspondence selection on CPU `FastGICP` / `FastVGICP`.

All of the new functionality is disabled by default. When disabled, the new code path keeps the original behavior of this branch.

## Feature support matrix

| Feature | FastGICP | FastVGICP | FastVGICP_CUDA | NDT_CUDA |
| --- | --- | --- | --- | --- |
| Hessian observability diagnostics | yes | yes | yes | yes |
| Auto soft suppression / hard DOF lock | yes | yes | yes | yes |
| Sparse 3D visual anchors | yes | yes | yes | yes |
| Dense RGB-assisted matching | yes | yes | no | no |

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

# optional RGB-assisted correspondence selection on CPU FastGICP / FastVGICP
gicp.set_source_colors(source_rgb)  # Nx3, float or uint8-like values
gicp.set_target_colors(target_rgb)
gicp.set_color_matching(True, candidate_count=5, color_weight=0.5, color_sigma=16.0)

# inspect the final Hessian diagnostics
diag = gicp.get_observability_diagnostics()
print(diag["estimated_rank"], diag["ambiguity_scores"])

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

```

## C++ API additions

Useful new entry points on `fast_gicp::LsqRegistration`:

- `setObservabilityCheck(bool)`
- `setObservabilityEigenThresholds(double relative, double absolute)`
- `setAutoSoftPriorStrength(double)`
- `setHardLockMask(std::array<int, 6>)`
- `setPreferredAmbiguousMask(std::array<int, 6>)`
- `setSparseAnchorCorrespondences(source_points, target_points, weights, sigmas)`
- `getObservabilityDiagnostics()`
- `getFinalRegularizedHessian()`
- `setAlignmentQualityConfig(...)`
- `getAlignmentQualityConfig()`
- `getAlignmentQualityReport()`

Useful new entry points on CPU `fast_gicp::FastGICP` / `fast_gicp::FastVGICP`:

- `setSourceColors(...)`
- `setTargetColors(...)`
- `setColorMatchingConfig(...)`

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
    - observability: `rank`, `condition_number`, `ambiguity_scores`, `max_ambiguity`
    - anchor diagnostics: `anchor_count`, `anchor_mean_residual`, `anchor_p95_residual`
    - advisory result: `suggested_accept`, `rejection_reasons`
  - DOF order is always `[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]`.
  - Recommended SLAM use:
    - do not trust `converged` alone
    - reject or downweight alignments with high tail residuals or high ambiguity on critical DOFs such as `trans_z` or yaw
    - Set `color_sigma` to match your RGB numeric range; too small makes color overly sharp, too large makes it ineffective.

### Practical tuning order

1. Keep all new features off and confirm the baseline behavior.
2. Enable observability check and diagnostics.
3. Add a small `auto_soft_prior_strength`.
4. Set `preferred_ambiguous_mask` for the DOFs you want suppressed first.
5. If drift remains, add sparse anchors.
6. If the scene has useful texture/color but weak geometry, enable dense RGB correspondence.
7. Only use hard lock for persistent, well-understood failure modes.

Recommended policy for simple vertical-plane scenes:

- Enable observability check.
- Set a small `auto_soft_prior_strength`, for example `0.01 ~ 0.05`.
- Mark `trans_z` in `preferred_ambiguous_mask`.
- If your application requires strict suppression, hard-lock `trans_z`.
- If you have reliable visual matches, add sparse anchors before alignment.

## Build and test

Build with Python bindings, self-checks, and benchmark:

```bash
source /nvme1/jiaheng/miniconda3/etc/profile.d/conda.sh
conda activate nsl2
cmake -S . -B cmake-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_apps=OFF \
  -DBUILD_test=ON \
  -DBUILD_benchmarks=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPCL_DIR=$CONDA_PREFIX/share/pcl-1.14 \
  -DBOOST_ROOT=$CONDA_PREFIX \
  -Dpybind11_DIR=$CONDA_PREFIX/lib/python3.11/site-packages/pybind11/share/cmake/pybind11
cmake --build cmake-build -j4
```

Run the built-in self-check:

```bash
./cmake-build/gicp_selfcheck
```

Run the speed benchmark against the original no-extension baseline of this branch
(that is: all new features disabled) and compare it with the robust configurations:

```bash
./cmake-build/gicp_benchmark ./data 20
```

The benchmark prints `baseline_default`, `robust_observability`, and `robust_observability_anchor`.
`baseline_default` is the reference for the original behavior because every new feature remains disabled there.
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

