// FastGICPCudaCore — Phase B2 implementation
//
// Kernels: brute-force KNN, per-point covariance with regularization,
// per-iteration correspondence + Mahalanobis residual + H/b reduction.
//
// All kernels live in this single TU to keep the build graph small. The
// header (`fast_gicp_cuda_core.cuh`) hides thrust types via PImpl.

#include <fast_gicp/cuda/fast_gicp_cuda_core.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <cuda_runtime.h>

#ifdef USE_CUVS
// cuVS C-API for ANN/KNN search. Brute-force is used because: (a) target
// changes every align(), index-built methods (IVF/HNSW) would rebuild every
// frame which dominates KNN time; (b) cuVS brute-force exploits cuBLAS GEMM
// for pairwise L2 + a warp-cooperative top-k, giving ~10-30x over our hand-
// rolled kernel on N>100K.
#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/common.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define FG_CUDA_CHECK(call) do { \
  cudaError_t _err = (call); \
  if (_err != cudaSuccess) { \
    std::fprintf(stderr, "[fast_gicp_cuda] %s:%d %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(_err)); \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err)); \
  } \
} while(0)

namespace fast_gicp {
namespace cuda {

constexpr int kMaxK = 32;  // upper bound on k_correspondences (in-register heap size)

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
struct FastGICPCudaCoreState {
  thrust::device_vector<Eigen::Vector3f> source_points;
  thrust::device_vector<Eigen::Vector3f> target_points;
#ifdef USE_CUVS
  // cuVS resources (cuBLAS/raft handle) cached across launches.
  // Creating cuvsResources is expensive (cuBLAS handle init etc.), so reuse
  // across linearize() / covariance build calls.
  cuvsResources_t cuvs_res = 0;
  bool cuvs_res_init = false;
#endif

  // Source/target per-point 3x3 covariance (already regularized).
  thrust::device_vector<Eigen::Matrix3f> source_covs;
  thrust::device_vector<Eigen::Matrix3f> target_covs;

  // KNN of source-against-source and target-against-target (k_correspondences
  // neighbors per point). Computed once when covariances need to be built.
  thrust::device_vector<int> source_knn_idx;        // size = N_s * k
  thrust::device_vector<int> target_knn_idx;        // size = N_t * k

  // Per-iteration correspondence cache (source -> target).
  thrust::device_vector<int> correspondences;       // size = N_s; -1 if none
  thrust::device_vector<float> sq_distances;        // size = N_s
  thrust::device_vector<Eigen::Matrix3f> mahalanobis;  // size = N_s

  // Per-source residual squared (Mahalanobis r^2).
  thrust::device_vector<float> correspondence_residuals;
  // Runtime weights (mirrored from host).
  thrust::device_vector<float> correspondence_weights;

  // Sparse anchors
  thrust::device_vector<Eigen::Vector3f> anchor_source;
  thrust::device_vector<Eigen::Vector3f> anchor_target;
  thrust::device_vector<float> anchor_weight_base;
  thrust::device_vector<float> anchor_sigma;
  thrust::device_vector<float> anchor_runtime_weight;
  thrust::device_vector<float> anchor_residuals;

  // 2DGS raw input retention + per-side mode/ratio cached so the GPU build
  // can rerun if marked dirty.
  thrust::device_vector<float> source_rotationsq_xyzw;
  thrust::device_vector<float> source_scales_2d;
  thrust::device_vector<float> target_rotationsq_xyzw;
  thrust::device_vector<float> target_scales_2d;
  int source_2dgs_mode = -1;     // -1: no 2DGS input
  int target_2dgs_mode = -1;
  float source_2dgs_normal_ratio = 0.05f;
  float source_2dgs_normal_min = 1e-4f;
  float target_2dgs_normal_ratio = 0.05f;
  float target_2dgs_normal_min = 1e-4f;

  // Cache flags
  bool source_covs_dirty = true;
  bool target_covs_dirty = true;

  bool use_sparse_anchors = false;
};

// ---------------------------------------------------------------------------
// KNN kernel — brute force, 1 thread per query, in-register max-heap of size k.
// Heap layout: array of size k holding (sq_dist, idx) pairs; index 0 is the
// largest distance (max-heap). We only swap with top if the new distance is
// smaller. At the end we have the k smallest distances (unsorted).
// ---------------------------------------------------------------------------
__device__ inline void sift_down(float* d, int* idx, int n, int root) {
  int cur = root;
  while (true) {
    int l = 2 * cur + 1;
    int r = 2 * cur + 2;
    int largest = cur;
    if (l < n && d[l] > d[largest]) largest = l;
    if (r < n && d[r] > d[largest]) largest = r;
    if (largest == cur) break;
    float tf = d[cur]; d[cur] = d[largest]; d[largest] = tf;
    int ti = idx[cur]; idx[cur] = idx[largest]; idx[largest] = ti;
    cur = largest;
  }
}

__global__ void knn_brute_force_kernel(
  const Eigen::Vector3f* __restrict__ query,
  int num_query,
  const Eigen::Vector3f* __restrict__ targets,
  int num_targets,
  int k,
  int* __restrict__ out_indices,           // num_query * k
  float* __restrict__ out_sq_dists,         // num_query * k
  int self_skip_offset                       // -1 to disable; otherwise skip
                                              // matching index equal to query
                                              // index + offset (for self-KNN)
) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_query) return;
  const Eigen::Vector3f x = query[q];

  float d[kMaxK];
  int idx[kMaxK];

  // Fill the heap with the first `k` valid candidates.
  int filled = 0;
  int t = 0;
  while (filled < k && t < num_targets) {
    if (self_skip_offset >= 0 && t == q + self_skip_offset) { ++t; continue; }
    const Eigen::Vector3f diff = targets[t] - x;
    d[filled] = diff.squaredNorm();
    idx[filled] = t;
    ++filled;
    ++t;
  }
  // If fewer than k targets exist, pad with sentinels.
  for (int i = filled; i < k; ++i) { d[i] = 1e30f; idx[i] = -1; }

  // Build max-heap on d[0..k-1].
  for (int root = (k - 2) / 2; root >= 0; --root) sift_down(d, idx, k, root);

  // Process remaining targets.
  for (; t < num_targets; ++t) {
    if (self_skip_offset >= 0 && t == q + self_skip_offset) continue;
    const Eigen::Vector3f diff = targets[t] - x;
    const float sd = diff.squaredNorm();
    if (sd < d[0]) {
      d[0] = sd;
      idx[0] = t;
      sift_down(d, idx, k, 0);
    }
  }

  // Write out
  for (int i = 0; i < k; ++i) {
    out_indices[q * k + i] = idx[i];
    out_sq_dists[q * k + i] = d[i];
  }
}

// ---------------------------------------------------------------------------
// Covariance kernel — given KNN indices for each point, compute 3x3 covariance
// of the K neighbors and regularize per the chosen scheme.
// ---------------------------------------------------------------------------
// ---- Hand-coded symmetric 3x3 eigendecomposition (Cardano) ---------------
// Eigenvalues ascending in `evals`, corresponding columns of `evecs` are unit
// eigenvectors. Used because Eigen::SelfAdjointEigenSolver pulls in host-only
// code on device (NVCC warns "calling a __host__ function ... not allowed").
//
// References: Smith 1961; Kopp 2008. Stable for ill-conditioned matrices via
// shift to trace-zero form.
__device__ inline void sym_eig3(const Eigen::Matrix3f& A,
                                Eigen::Vector3f& evals,
                                Eigen::Matrix3f& evecs) {
  // Use double internally for stability; outputs are float.
  const double a00 = A(0,0), a01 = A(0,1), a02 = A(0,2);
  const double a11 = A(1,1), a12 = A(1,2), a22 = A(2,2);

  // Trace and shifted form B = A - (tr/3)*I -> eigenvalues sum to 0.
  const double q = (a00 + a11 + a22) / 3.0;
  const double b00 = a00 - q;
  const double b11 = a11 - q;
  const double b22 = a22 - q;

  // p2 = ||B||_F^2 / 6 (Smith's parameter)
  const double p2 = (b00*b00 + b11*b11 + b22*b22 + 2.0*(a01*a01 + a02*a02 + a12*a12)) / 6.0;
  double e0, e1, e2;
  if (p2 < 1e-30) {
    // Already diagonal & isotropic
    e0 = e1 = e2 = q;
  } else {
    const double p = sqrt(p2);
    // det(B/p)/2
    const double inv_p = 1.0 / p;
    const double c00 = b00 * inv_p, c11 = b11 * inv_p, c22 = b22 * inv_p;
    const double c01 = a01 * inv_p, c02 = a02 * inv_p, c12 = a12 * inv_p;
    const double det_c = c00*(c11*c22 - c12*c12) - c01*(c01*c22 - c12*c02) + c02*(c01*c12 - c11*c02);
    double r = det_c * 0.5;
    if (r < -1.0) r = -1.0;
    if (r >  1.0) r =  1.0;
    const double phi = acos(r) / 3.0;
    e0 = q + 2.0 * p * cos(phi);                 // largest
    e2 = q + 2.0 * p * cos(phi + 2.0944);        // smallest (2*pi/3)
    e1 = 3.0 * q - e0 - e2;                       // middle
  }
  // sort ascending: e2 <= e1 <= e0
  evals(0) = (float)e2;
  evals(1) = (float)e1;
  evals(2) = (float)e0;

  // Eigenvector for a given eigenvalue lambda: find a non-zero vector in the
  // null space of (A - lambda*I). We compute the cross product of two rows;
  // pick the largest cross result for numerical stability.
  auto eigvec_for = [&](float lam) -> Eigen::Vector3f {
    Eigen::Matrix3f M = A;
    M(0,0) -= lam; M(1,1) -= lam; M(2,2) -= lam;
    Eigen::Vector3f r0 = M.row(0);
    Eigen::Vector3f r1 = M.row(1);
    Eigen::Vector3f r2 = M.row(2);
    Eigen::Vector3f v01 = r0.cross(r1);
    Eigen::Vector3f v02 = r0.cross(r2);
    Eigen::Vector3f v12 = r1.cross(r2);
    float n01 = v01.squaredNorm();
    float n02 = v02.squaredNorm();
    float n12 = v12.squaredNorm();
    Eigen::Vector3f v;
    if (n01 >= n02 && n01 >= n12) v = v01;
    else if (n02 >= n12) v = v02;
    else v = v12;
    float n = v.norm();
    if (n < 1e-12f) {
      // Degenerate (repeated eigenvalue); pick a fallback orthogonal axis.
      v = Eigen::Vector3f::UnitZ();
    } else {
      v /= n;
    }
    return v;
  };

  Eigen::Vector3f v0 = eigvec_for(evals(0));
  Eigen::Vector3f v2 = eigvec_for(evals(2));
  // Make v2 orthogonal to v0 (Gram-Schmidt) to handle near-repeated eigenvalues.
  v2 -= v2.dot(v0) * v0;
  float nv2 = v2.norm();
  if (nv2 < 1e-12f) {
    // Fallback: any vector orthogonal to v0
    if (fabs(v0(0)) < 0.9f) v2 = Eigen::Vector3f::UnitX();
    else v2 = Eigen::Vector3f::UnitY();
    v2 -= v2.dot(v0) * v0;
    v2 /= v2.norm();
  } else {
    v2 /= nv2;
  }
  Eigen::Vector3f v1 = v2.cross(v0);
  evecs.col(0) = v0;
  evecs.col(1) = v1;
  evecs.col(2) = v2;
}

__device__ inline void regularize_plane(Eigen::Matrix3f& cov) {
  Eigen::Vector3f evals;
  Eigen::Matrix3f evecs;
  sym_eig3(cov, evals, evecs);
  // ascending evals; "plane" sets smallest = 1e-3, others = 1.
  Eigen::Vector3f vals; vals << 1e-3f, 1.f, 1.f;
  cov = evecs * vals.asDiagonal() * evecs.transpose();
}

__device__ inline void regularize_min_eig(Eigen::Matrix3f& cov) {
  Eigen::Vector3f evals;
  Eigen::Matrix3f evecs;
  sym_eig3(cov, evals, evecs);
  Eigen::Vector3f vals;
  vals(0) = fmaxf(evals(0), 1e-3f);
  vals(1) = fmaxf(evals(1), 1e-3f);
  vals(2) = fmaxf(evals(2), 1e-3f);
  cov = evecs * vals.asDiagonal() * evecs.transpose();
}

__device__ inline void regularize_frobenius(Eigen::Matrix3f& cov) {
  Eigen::Matrix3f C = cov + 1e-3f * Eigen::Matrix3f::Identity();
  Eigen::Matrix3f Ci = C.inverse();
  cov = (Ci / Ci.norm()).inverse();
}

// ---- 2DGS surfel covariance kernel ----------------------------------------
// Each surfel: (quaternion R, 2D in-plane scales s1, s2). Builds a 3x3 disc-
// like covariance in world frame: R * diag(svals) * R^T.
// mode 0 ("physical"): svals = (s1^2, s2^2, sn^2)
// mode 1 ("normalized"): svals = (1, 1, 1e-3)
__device__ inline Eigen::Matrix3f quat_xyzw_to_rotation(float x, float y, float z, float w) {
  // Normalize defensively.
  const float n = sqrtf(x*x + y*y + z*z + w*w);
  if (n < 1e-12f) {
    return Eigen::Matrix3f::Identity();
  }
  const float inv = 1.0f / n;
  x *= inv; y *= inv; z *= inv; w *= inv;
  Eigen::Matrix3f R;
  R(0,0) = 1.f - 2.f*(y*y + z*z);
  R(0,1) = 2.f*(x*y - z*w);
  R(0,2) = 2.f*(x*z + y*w);
  R(1,0) = 2.f*(x*y + z*w);
  R(1,1) = 1.f - 2.f*(x*x + z*z);
  R(1,2) = 2.f*(y*z - x*w);
  R(2,0) = 2.f*(x*z - y*w);
  R(2,1) = 2.f*(y*z + x*w);
  R(2,2) = 1.f - 2.f*(x*x + y*y);
  return R;
}

__global__ void compute_2dgs_covariances_kernel(
  const float* __restrict__ rotations_xyzw,  // 4*N
  const float* __restrict__ scales_2d,        // 2*N
  int n,
  int mode_int,                                // 0=physical, 1=normalized
  float normal_sigma_ratio,
  float normal_sigma_min,
  Eigen::Matrix3f* __restrict__ out_covs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float s1 = fmaxf(scales_2d[2 * i + 0], 1e-12f);
  const float s2 = fmaxf(scales_2d[2 * i + 1], 1e-12f);
  const float sn = fmaxf(normal_sigma_ratio * fminf(s1, s2), normal_sigma_min);

  const float x = rotations_xyzw[4 * i + 0];
  const float y = rotations_xyzw[4 * i + 1];
  const float z = rotations_xyzw[4 * i + 2];
  const float w = rotations_xyzw[4 * i + 3];
  const Eigen::Matrix3f R = quat_xyzw_to_rotation(x, y, z, w);

  Eigen::Vector3f svals;
  if (mode_int == 1) {
    svals << 1.f, 1.f, 1e-3f;
  } else {
    svals << s1 * s1, s2 * s2, sn * sn;
  }
  out_covs[i] = R * svals.asDiagonal() * R.transpose();
}

__global__ void compute_covariances_kernel(
  const Eigen::Vector3f* __restrict__ points,
  int num_points,
  const int* __restrict__ knn_idx,
  int k,
  int regularization_method,
  Eigen::Matrix3f* __restrict__ out_covs) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_points) return;

  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  Eigen::Vector3f mean = Eigen::Vector3f::Zero();
  int n = 0;
  for (int i = 0; i < k; ++i) {
    const int ni = knn_idx[q * k + i];
    if (ni < 0) continue;
    const Eigen::Vector3f p = points[ni];
    mean += p;
    ++n;
  }
  if (n < 2) {
    out_covs[q] = Eigen::Matrix3f::Identity() * 1e-3f;
    return;
  }
  mean /= static_cast<float>(n);
  for (int i = 0; i < k; ++i) {
    const int ni = knn_idx[q * k + i];
    if (ni < 0) continue;
    const Eigen::Vector3f d = points[ni] - mean;
    cov += d * d.transpose();
  }
  cov /= static_cast<float>(n);

  switch (regularization_method) {
    case 0:  // NONE
      break;
    case 1:  // MIN_EIG
    case 2:  // NORMALIZED_MIN_EIG
      regularize_min_eig(cov);
      break;
    case 3:  // PLANE
      regularize_plane(cov);
      break;
    case 4:  // FROBENIUS
      regularize_frobenius(cov);
      break;
    case 5:  // NORMALIZED_ELLIPSE
      regularize_min_eig(cov);  // approximation; CPU does a fancier rescale
      break;
    default:
      regularize_plane(cov);
      break;
  }
  // Ensure symmetry
  cov = 0.5f * (cov + cov.transpose());
  out_covs[q] = cov;
}

// ---------------------------------------------------------------------------
// Linearize kernel — for each source point i:
//   1) transform: x_i = T * src_i
//   2) brute-force 1-NN among target points within max_corr_dist
//   3) compute Mahalanobis = (cov_b + R cov_a R^T)^{-1}
//   4) residual r = mean_b - x_i; r^2 = r^T M r
//   5) Jacobian J = [skew(x_i) | -I]
//   6) Hi = w * J^T M J, bi = w * J^T M r
//   7) atomicAdd Hi/bi into global 6x6 H and 6x1 b (doubles for stability)
// ---------------------------------------------------------------------------
struct PoseT {
  Eigen::Matrix3f R;
  Eigen::Vector3f t;
};

__device__ inline Eigen::Matrix3f skew(const Eigen::Vector3f& v) {
  Eigen::Matrix3f S;
  S <<     0, -v.z(),  v.y(),
       v.z(),      0, -v.x(),
      -v.y(),  v.x(),      0;
  return S;
}

__device__ inline int nearest_target_brute(
  const Eigen::Vector3f& x,
  const Eigen::Vector3f* targets, int n_t,
  float max_sq_dist, float* out_sq_dist) {
  float best = max_sq_dist;
  int best_idx = -1;
  for (int t = 0; t < n_t; ++t) {
    const Eigen::Vector3f d = targets[t] - x;
    const float sd = d.squaredNorm();
    if (sd < best) {
      best = sd;
      best_idx = t;
    }
  }
  *out_sq_dist = best;
  return best_idx;
}

__global__ void linearize_kernel(
  // inputs
  const Eigen::Vector3f* __restrict__ src,
  const Eigen::Vector3f* __restrict__ tgt,
  const Eigen::Matrix3f* __restrict__ src_covs,
  const Eigen::Matrix3f* __restrict__ tgt_covs,
  int n_s, int n_t,
  PoseT T,
  float max_corr_sq_dist,
  const float* __restrict__ runtime_weights,  // may be nullptr
  int use_runtime_weights,
  // outputs (per-source)
  int* __restrict__ out_corr,                  // -1 if no match
  float* __restrict__ out_sq_dist,
  Eigen::Matrix3f* __restrict__ out_mahal,
  float* __restrict__ out_residual,            // Mahalanobis r^2
  // outputs (reduced atomic)
  double* __restrict__ H_global,               // 36 doubles
  double* __restrict__ b_global,               // 6 doubles
  double* __restrict__ cost_global             // 1 double
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_s) return;

  const Eigen::Vector3f mean_a = src[i];
  const Eigen::Vector3f x_i = T.R * mean_a + T.t;

  // 1-NN
  float sd = 0.0f;
  int j = nearest_target_brute(x_i, tgt, n_t, max_corr_sq_dist, &sd);
  out_corr[i] = j;
  out_sq_dist[i] = sd;
  if (j < 0) {
    out_residual[i] = -1.0f;
    return;
  }

  // Combined covariance and Mahalanobis
  const Eigen::Matrix3f cov_a = src_covs[i];
  const Eigen::Matrix3f cov_b = tgt_covs[j];
  const Eigen::Matrix3f RCRt = T.R * cov_a * T.R.transpose();
  Eigen::Matrix3f RCR = cov_b + RCRt;
  // Stabilise the inverse: tiny ridge if near-singular.
  // (Most well-formed inputs will have RCR strictly SPD.)
  RCR += 1e-9f * Eigen::Matrix3f::Identity();
  const Eigen::Matrix3f M = RCR.inverse();
  out_mahal[i] = M;

  const Eigen::Vector3f mean_b = tgt[j];
  const Eigen::Vector3f r = mean_b - x_i;
  const float r2 = (r.transpose() * M * r)(0, 0);
  out_residual[i] = r2;

  const float w = use_runtime_weights ? max(0.f, min(1.f, runtime_weights[i])) : 1.f;
  if (w == 0.f) return;

  // Jacobian J (3x6): [ skew(x_i) | -I ]
  Eigen::Matrix<float, 3, 6> J;
  J.block<3, 3>(0, 0) = skew(x_i);
  J.block<3, 3>(0, 3) = -Eigen::Matrix3f::Identity();

  // Hi = J^T M J, bi = J^T M r
  const Eigen::Matrix<float, 6, 6> Hi = J.transpose() * M * J;
  const Eigen::Matrix<float, 6, 1> bi = J.transpose() * M * r;

  // atomicAdd into global H/b/cost in double precision.
  for (int rr = 0; rr < 6; ++rr) {
    for (int cc = 0; cc < 6; ++cc) {
      atomicAdd(&H_global[rr * 6 + cc], static_cast<double>(w * Hi(rr, cc)));
    }
    atomicAdd(&b_global[rr], static_cast<double>(w * bi(rr)));
  }
  atomicAdd(cost_global, static_cast<double>(w * r2));
}

// Cost-only kernel (no H/b, no correspondence search — uses cached corr/mahal)
__global__ void cost_kernel(
  const Eigen::Vector3f* __restrict__ src,
  const Eigen::Vector3f* __restrict__ tgt,
  const Eigen::Matrix3f* __restrict__ mahal,
  const int* __restrict__ corr,
  int n_s,
  PoseT T,
  const float* __restrict__ runtime_weights,
  int use_runtime_weights,
  double* __restrict__ cost_global) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_s) return;
  const int j = corr[i];
  if (j < 0) return;
  const Eigen::Vector3f x_i = T.R * src[i] + T.t;
  const Eigen::Vector3f r = tgt[j] - x_i;
  const Eigen::Matrix3f& M = mahal[i];
  const float r2 = (r.transpose() * M * r)(0, 0);
  const float w = use_runtime_weights ? max(0.f, min(1.f, runtime_weights[i])) : 1.f;
  atomicAdd(cost_global, static_cast<double>(w * r2));
}

// ---------------------------------------------------------------------------
// FastGICPCudaCore implementation
// ---------------------------------------------------------------------------

FastGICPCudaCore::FastGICPCudaCore() : state_(new FastGICPCudaCoreState()) {
  cudaDeviceSynchronize();
}
FastGICPCudaCore::~FastGICPCudaCore() {
#ifdef USE_CUVS
  if (state_ && state_->cuvs_res_init) {
    cuvsResourcesDestroy(state_->cuvs_res);
    state_->cuvs_res_init = false;
  }
#endif
}

void FastGICPCudaCore::set_correspondence_randomness(int k) {
  if (k < 1 || k > kMaxK) {
    throw std::invalid_argument("FastGICPCuda: k_correspondences must be in [1, 32]");
  }
  k_correspondences_ = k;
  state_->source_covs_dirty = true;
  state_->target_covs_dirty = true;
}
void FastGICPCudaCore::set_regularization_method(RegularizationMethod m) {
  regularization_method_ = m;
  state_->source_covs_dirty = true;
  state_->target_covs_dirty = true;
}
void FastGICPCudaCore::set_max_correspondence_distance(double d) {
  max_correspondence_distance_ = d;
}
void FastGICPCudaCore::set_num_threads_hint(int n) { num_threads_hint_ = n; }

void FastGICPCudaCore::set_knn_backend(const std::string& backend) {
  if (backend == "brute_force") { knn_backend_ = backend; return; }
#ifdef USE_CUVS
  if (backend == "cuvs") { knn_backend_ = backend; return; }
#endif
  throw std::runtime_error("FastGICPCuda: unknown knn backend '" + backend +
                           "' (compile with -DUSE_CUVS=ON for the 'cuvs' option)");
}

void FastGICPCudaCore::set_dynamic_rejection_config(const DynamicRejectionConfig& cfg) {
  dyn_cfg_ = cfg;
}

// ---- Data ingestion -------------------------------------------------------
namespace {
void copy_to_device(const float* host_xyz, int count, thrust::device_vector<Eigen::Vector3f>* dst) {
  dst->resize(count);
  if (count == 0) return;
  const Eigen::Vector3f* typed = reinterpret_cast<const Eigen::Vector3f*>(host_xyz);
  thrust::copy(typed, typed + count, dst->begin());
}
}  // namespace

void FastGICPCudaCore::set_source_points(const float* host_points, int count) {
  copy_to_device(host_points, count, &state_->source_points);
  state_->source_covs.clear();
  state_->source_knn_idx.clear();
  state_->source_covs_dirty = true;
  state_->source_rotationsq_xyzw.clear();
  state_->source_scales_2d.clear();
  state_->correspondences.clear();
  state_->sq_distances.clear();
  state_->mahalanobis.clear();
  state_->correspondence_residuals.clear();
  state_->correspondence_weights.clear();
}
void FastGICPCudaCore::set_target_points(const float* host_points, int count) {
  copy_to_device(host_points, count, &state_->target_points);
  state_->target_covs.clear();
  state_->target_knn_idx.clear();
  state_->target_covs_dirty = true;
  state_->target_rotationsq_xyzw.clear();
  state_->target_scales_2d.clear();
}
void FastGICPCudaCore::swap_source_target() {
  state_->source_points.swap(state_->target_points);
  state_->source_covs.swap(state_->target_covs);
  state_->source_knn_idx.swap(state_->target_knn_idx);
  state_->source_rotationsq_xyzw.swap(state_->target_rotationsq_xyzw);
  state_->source_scales_2d.swap(state_->target_scales_2d);
  state_->source_covs_dirty = true;  // sizes match but mapping changed
  state_->target_covs_dirty = true;
  state_->correspondences.clear();
}
void FastGICPCudaCore::clear_source() {
  state_->source_points.clear(); state_->source_covs.clear();
  state_->source_knn_idx.clear(); state_->source_covs_dirty = true;
  state_->correspondences.clear(); state_->sq_distances.clear();
  state_->mahalanobis.clear(); state_->correspondence_residuals.clear();
  state_->correspondence_weights.clear();
  state_->source_rotationsq_xyzw.clear(); state_->source_scales_2d.clear();
}
void FastGICPCudaCore::clear_target() {
  state_->target_points.clear(); state_->target_covs.clear();
  state_->target_knn_idx.clear(); state_->target_covs_dirty = true;
  state_->target_rotationsq_xyzw.clear(); state_->target_scales_2d.clear();
}
int FastGICPCudaCore::source_size() const { return static_cast<int>(state_->source_points.size()); }
int FastGICPCudaCore::target_size() const { return static_cast<int>(state_->target_points.size()); }

// ---- 2DGS / anchors / weights ---------------------------------------------
namespace {
int parse_2dgs_mode(const std::string& mode) {
  if (mode == "physical") return 0;
  if (mode == "normalized") return 1;
  throw std::invalid_argument("FastGICPCuda: unknown 2DGS mode '" + mode + "'");
}
}  // namespace

void FastGICPCudaCore::set_source_covariances_2dgs(
  const float* rotations_xyzw, const float* scales_2d, int count,
  const std::string& mode, double normal_sigma_ratio, double normal_sigma_min) {
  state_->source_rotationsq_xyzw.assign(rotations_xyzw, rotations_xyzw + count * 4);
  state_->source_scales_2d.assign(scales_2d, scales_2d + count * 2);
  state_->source_2dgs_mode = parse_2dgs_mode(mode);
  state_->source_2dgs_normal_ratio = static_cast<float>(normal_sigma_ratio);
  state_->source_2dgs_normal_min = static_cast<float>(normal_sigma_min);
  state_->source_covs_dirty = true;
}
void FastGICPCudaCore::set_target_covariances_2dgs(
  const float* rotations_xyzw, const float* scales_2d, int count,
  const std::string& mode, double normal_sigma_ratio, double normal_sigma_min) {
  state_->target_rotationsq_xyzw.assign(rotations_xyzw, rotations_xyzw + count * 4);
  state_->target_scales_2d.assign(scales_2d, scales_2d + count * 2);
  state_->target_2dgs_mode = parse_2dgs_mode(mode);
  state_->target_2dgs_normal_ratio = static_cast<float>(normal_sigma_ratio);
  state_->target_2dgs_normal_min = static_cast<float>(normal_sigma_min);
  state_->target_covs_dirty = true;
}

void FastGICPCudaCore::set_sparse_anchor_correspondences(
  const double* source_points, const double* target_points, int count,
  const double* weights, int weights_count,
  const double* sigmas, int sigmas_count) {
  thrust::host_vector<Eigen::Vector3f> hs(count), ht(count);
  for (int i = 0; i < count; ++i) {
    hs[i] = Eigen::Vector3f((float)source_points[3*i+0], (float)source_points[3*i+1], (float)source_points[3*i+2]);
    ht[i] = Eigen::Vector3f((float)target_points[3*i+0], (float)target_points[3*i+1], (float)target_points[3*i+2]);
  }
  state_->anchor_source = hs;
  state_->anchor_target = ht;
  thrust::host_vector<float> w(count, 1.f), sigma(count, 1.f);
  if (weights_count == count) for (int i = 0; i < count; ++i) w[i] = (float)weights[i];
  if (sigmas_count == count) for (int i = 0; i < count; ++i) sigma[i] = (float)std::max(sigmas[i], 1e-9);
  state_->anchor_weight_base = w;
  state_->anchor_sigma = sigma;
  state_->anchor_runtime_weight.assign(count, 1.f);
  state_->anchor_residuals.assign(count, 0.f);
  state_->use_sparse_anchors = (count > 0);
}
void FastGICPCudaCore::clear_sparse_anchor_correspondences() {
  state_->anchor_source.clear(); state_->anchor_target.clear();
  state_->anchor_weight_base.clear(); state_->anchor_sigma.clear();
  state_->anchor_runtime_weight.clear(); state_->anchor_residuals.clear();
  state_->use_sparse_anchors = false;
}
void FastGICPCudaCore::set_use_sparse_anchors(bool enable) { state_->use_sparse_anchors = enable; }

// ---- Iteration state ------------------------------------------------------
void FastGICPCudaCore::reset_iteration_state() {
  dyn_diag_ = DynamicRejectionDiagnostics();
  dyn_diag_.enabled = dyn_cfg_.enable && dyn_cfg_.kernel != DynamicRejectionKernel::NONE;
  gnc_mu_current_ = 0.0;
  gnc_mu_floor_ = 0.0;
  correspondence_residuals_host_.clear();
  correspondence_weights_host_.clear();
  anchor_residuals_host_.clear();
  anchor_weights_host_.clear();
  state_->correspondence_weights.clear();
  state_->anchor_runtime_weight.clear();
}

void FastGICPCudaCore::prepare_dynamic_weights_for_iteration(int /*iter*/) {
  // Phase B4 fills this.
}

// ---- Covariance build orchestration ---------------------------------------
namespace {

int regularization_to_int(RegularizationMethod m) {
  switch (m) {
    case RegularizationMethod::NONE: return 0;
    case RegularizationMethod::MIN_EIG: return 1;
    case RegularizationMethod::NORMALIZED_MIN_EIG: return 2;
    case RegularizationMethod::PLANE: return 3;
    case RegularizationMethod::FROBENIUS: return 4;
    case RegularizationMethod::NORMALIZED_ELLIPSE: return 5;
    default: return 3;  // PLANE
  }
}

void launch_knn_brute_force(
  const thrust::device_vector<Eigen::Vector3f>& query,
  const thrust::device_vector<Eigen::Vector3f>& targets,
  int k,
  thrust::device_vector<int>* out_idx,
  thrust::device_vector<float>* out_sq,
  bool self_knn) {
  const int n_q = static_cast<int>(query.size());
  const int n_t = static_cast<int>(targets.size());
  out_idx->assign(n_q * k, -1);
  out_sq->assign(n_q * k, 1e30f);
  if (n_q == 0 || n_t == 0) return;
  const int block = 128;
  const int grid = (n_q + block - 1) / block;
  knn_brute_force_kernel<<<grid, block>>>(
    thrust::raw_pointer_cast(query.data()), n_q,
    thrust::raw_pointer_cast(targets.data()), n_t,
    k,
    thrust::raw_pointer_cast(out_idx->data()),
    thrust::raw_pointer_cast(out_sq->data()),
    self_knn ? 0 : -1);
  FG_CUDA_CHECK(cudaGetLastError());
}

#ifdef USE_CUVS
namespace {
// cuVS brute_force returns int64 neighbour indices (verified against the
// cuvs python binding on 26.04). We allocate a side buffer and narrow to int
// after the search.
__global__ void cuvs_drop_self_kernel(const int64_t* __restrict__ in_idx_kp1,
                                       const float* __restrict__ in_sq_kp1,
                                       int n_q, int k,
                                       int* __restrict__ out_idx,
                                       float* __restrict__ out_sq) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= n_q) return;
  const int kp1 = k + 1;
  int written = 0;
  bool skipped = false;
  for (int i = 0; i < kp1 && written < k; ++i) {
    const int64_t nbr = in_idx_kp1[q * kp1 + i];
    if (!skipped && static_cast<int>(nbr) == q) {
      skipped = true;
      continue;
    }
    out_idx[q * k + written] = static_cast<int>(nbr);
    out_sq[q * k + written] = in_sq_kp1[q * kp1 + i];
    ++written;
  }
  if (written < k) {
    out_idx[q * k + written] = static_cast<int>(in_idx_kp1[q * kp1 + k]);
    out_sq[q * k + written] = in_sq_kp1[q * kp1 + k];
  }
}

__global__ void cuvs_copy_idx_kernel(const int64_t* __restrict__ in_idx,
                                      int n, int* __restrict__ out_idx) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out_idx[i] = static_cast<int>(in_idx[i]);
}
}  // namespace

// cuVS brute-force KNN. Returns squared L2 distances and indices for the
// top-k nearest neighbours in `targets` for every point in `query`. When
// `self_knn=true`, we ask cuVS for k+1 neighbours and drop the self-match
// per row in a post-processing kernel.
//
// `res` is the cached resources handle owned by FastGICPCudaCoreState.
void launch_knn_cuvs(
  cuvsResources_t res,
  const thrust::device_vector<Eigen::Vector3f>& query,
  const thrust::device_vector<Eigen::Vector3f>& targets,
  int k,
  thrust::device_vector<int>* out_idx,
  thrust::device_vector<float>* out_sq,
  bool self_knn) {
  const int n_q = static_cast<int>(query.size());
  const int n_t = static_cast<int>(targets.size());
  out_idx->assign(n_q * k, -1);
  out_sq->assign(n_q * k, 1e30f);
  if (n_q == 0 || n_t == 0) return;

  // Ask for k+1 when doing self-knn so we can drop the self hit.
  const int k_req = self_knn ? (k + 1) : k;

  cuvsError_t err;

  // Build the index over the targets. Eigen::Vector3f is 3 contiguous floats,
  // so an Nx3 row-major view of targets.data() is valid.
  DLManagedTensor dataset_tensor{};
  dataset_tensor.dl_tensor.data = const_cast<Eigen::Vector3f*>(thrust::raw_pointer_cast(targets.data()));
  int64_t d_shape[2] = {static_cast<int64_t>(n_t), 3};
  dataset_tensor.dl_tensor.shape = d_shape;
  dataset_tensor.dl_tensor.ndim = 2;
  dataset_tensor.dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
  dataset_tensor.dl_tensor.device = DLDevice{kDLCUDA, 0};
  dataset_tensor.dl_tensor.strides = nullptr;
  dataset_tensor.dl_tensor.byte_offset = 0;

  cuvsBruteForceIndex_t index = nullptr;
  err = cuvsBruteForceIndexCreate(&index);
  if (err != CUVS_SUCCESS) {
    throw std::runtime_error(std::string("cuVS: cuvsBruteForceIndexCreate failed: ") +
                             (cuvsGetLastErrorText() ? cuvsGetLastErrorText() : ""));
  }
  err = cuvsBruteForceBuild(res, &dataset_tensor, L2Expanded, 0.0f, index);
  if (err != CUVS_SUCCESS) {
    cuvsBruteForceIndexDestroy(index);
    throw std::runtime_error(std::string("cuVS: cuvsBruteForceBuild failed: ") +
                             (cuvsGetLastErrorText() ? cuvsGetLastErrorText() : ""));
  }

  // Search tensors.
  DLManagedTensor q_tensor{};
  q_tensor.dl_tensor.data = const_cast<Eigen::Vector3f*>(thrust::raw_pointer_cast(query.data()));
  int64_t q_shape[2] = {static_cast<int64_t>(n_q), 3};
  q_tensor.dl_tensor.shape = q_shape;
  q_tensor.dl_tensor.ndim = 2;
  q_tensor.dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
  q_tensor.dl_tensor.device = DLDevice{kDLCUDA, 0};
  q_tensor.dl_tensor.strides = nullptr;
  q_tensor.dl_tensor.byte_offset = 0;

  // cuVS brute_force returns int64 neighbour indices (per cuvs 26.04 python
  // binding). Use an int64 scratch and narrow to int afterwards.
  thrust::device_vector<int64_t> idx_i64(static_cast<std::size_t>(n_q) * k_req);
  thrust::device_vector<float> dist_kp1;
  thrust::device_vector<float>* dist_buf = out_sq;
  if (self_knn) {
    dist_kp1.resize(static_cast<std::size_t>(n_q) * k_req);
    dist_buf = &dist_kp1;
  }

  DLManagedTensor n_tensor{};
  n_tensor.dl_tensor.data = thrust::raw_pointer_cast(idx_i64.data());
  int64_t k_shape[2] = {static_cast<int64_t>(n_q), static_cast<int64_t>(k_req)};
  n_tensor.dl_tensor.shape = k_shape;
  n_tensor.dl_tensor.ndim = 2;
  n_tensor.dl_tensor.dtype = DLDataType{kDLInt, 64, 1};
  n_tensor.dl_tensor.device = DLDevice{kDLCUDA, 0};
  n_tensor.dl_tensor.strides = nullptr;
  n_tensor.dl_tensor.byte_offset = 0;

  DLManagedTensor d_tensor{};
  d_tensor.dl_tensor.data = thrust::raw_pointer_cast(dist_buf->data());
  d_tensor.dl_tensor.shape = k_shape;
  d_tensor.dl_tensor.ndim = 2;
  d_tensor.dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
  d_tensor.dl_tensor.device = DLDevice{kDLCUDA, 0};
  d_tensor.dl_tensor.strides = nullptr;
  d_tensor.dl_tensor.byte_offset = 0;

  cuvsFilter prefilter;
  prefilter.addr = 0;
  prefilter.type = NO_FILTER;

  err = cuvsBruteForceSearch(res, index, &q_tensor, &n_tensor, &d_tensor, prefilter);
  cuvsBruteForceIndexDestroy(index);
  if (err != CUVS_SUCCESS) {
    throw std::runtime_error(std::string("cuVS: cuvsBruteForceSearch failed: ") +
                             (cuvsGetLastErrorText() ? cuvsGetLastErrorText() : ""));
  }

  const int block = 128;
  const int grid = (n_q + block - 1) / block;
  if (self_knn) {
    cuvs_drop_self_kernel<<<grid, block>>>(
      thrust::raw_pointer_cast(idx_i64.data()),
      thrust::raw_pointer_cast(dist_kp1.data()),
      n_q, k,
      thrust::raw_pointer_cast(out_idx->data()),
      thrust::raw_pointer_cast(out_sq->data()));
  } else {
    cuvs_copy_idx_kernel<<<(n_q * k + block - 1) / block, block>>>(
      thrust::raw_pointer_cast(idx_i64.data()),
      n_q * k,
      thrust::raw_pointer_cast(out_idx->data()));
    // distances already written directly into *out_sq above.
  }
  FG_CUDA_CHECK(cudaGetLastError());
}
#endif

void launch_knn(
  FastGICPCudaCoreState* state,
  const thrust::device_vector<Eigen::Vector3f>& query,
  const thrust::device_vector<Eigen::Vector3f>& targets,
  int k,
  thrust::device_vector<int>* out_idx,
  thrust::device_vector<float>* out_sq,
  bool self_knn,
  const std::string& backend) {
#ifdef USE_CUVS
  if (backend == "cuvs") {
    if (!state->cuvs_res_init) {
      if (cuvsResourcesCreate(&state->cuvs_res) != CUVS_SUCCESS) {
        throw std::runtime_error(std::string("cuVS: cuvsResourcesCreate failed: ") +
                                 (cuvsGetLastErrorText() ? cuvsGetLastErrorText() : ""));
      }
      state->cuvs_res_init = true;
    }
    launch_knn_cuvs(state->cuvs_res, query, targets, k, out_idx, out_sq, self_knn);
    return;
  }
#else
  (void)backend; (void)state;
#endif
  launch_knn_brute_force(query, targets, k, out_idx, out_sq, self_knn);
}

void launch_covariance(
  const thrust::device_vector<Eigen::Vector3f>& points,
  const thrust::device_vector<int>& knn_idx,
  int k,
  int regularization_int,
  thrust::device_vector<Eigen::Matrix3f>* out_covs) {
  const int n = static_cast<int>(points.size());
  out_covs->resize(n);
  if (n == 0) return;
  const int block = 128;
  const int grid = (n + block - 1) / block;
  compute_covariances_kernel<<<grid, block>>>(
    thrust::raw_pointer_cast(points.data()), n,
    thrust::raw_pointer_cast(knn_idx.data()), k,
    regularization_int,
    thrust::raw_pointer_cast(out_covs->data()));
  FG_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

namespace {
void launch_2dgs_covariance(
  const thrust::device_vector<float>& rotations_xyzw,
  const thrust::device_vector<float>& scales_2d,
  int mode_int, float normal_ratio, float normal_min,
  thrust::device_vector<Eigen::Matrix3f>* out_covs) {
  const int n = static_cast<int>(scales_2d.size() / 2);
  out_covs->resize(n);
  if (n == 0) return;
  const int block = 256;
  const int grid = (n + block - 1) / block;
  compute_2dgs_covariances_kernel<<<grid, block>>>(
    thrust::raw_pointer_cast(rotations_xyzw.data()),
    thrust::raw_pointer_cast(scales_2d.data()),
    n, mode_int, normal_ratio, normal_min,
    thrust::raw_pointer_cast(out_covs->data()));
  FG_CUDA_CHECK(cudaGetLastError());
}
}  // namespace

void FastGICPCudaCore::ensure_source_covariances_() {
  if (!state_->source_covs_dirty) return;
  const int n = static_cast<int>(state_->source_points.size());
  if (n == 0) return;
  // 2DGS path takes precedence when rotations+scales were provided and the
  // count matches the source point count.
  if (state_->source_2dgs_mode >= 0 &&
      static_cast<int>(state_->source_scales_2d.size() / 2) == n) {
    launch_2dgs_covariance(
      state_->source_rotationsq_xyzw, state_->source_scales_2d,
      state_->source_2dgs_mode, state_->source_2dgs_normal_ratio,
      state_->source_2dgs_normal_min, &state_->source_covs);
  } else {
    launch_knn(state_.get(),
               state_->source_points, state_->source_points, k_correspondences_,
               &state_->source_knn_idx, &state_->sq_distances, /*self_knn=*/true,
               knn_backend_);
    launch_covariance(state_->source_points, state_->source_knn_idx, k_correspondences_,
                      regularization_to_int(regularization_method_), &state_->source_covs);
  }
  state_->source_covs_dirty = false;
}

void FastGICPCudaCore::ensure_target_covariances_() {
  if (!state_->target_covs_dirty) return;
  const int n = static_cast<int>(state_->target_points.size());
  if (n == 0) return;
  if (state_->target_2dgs_mode >= 0 &&
      static_cast<int>(state_->target_scales_2d.size() / 2) == n) {
    launch_2dgs_covariance(
      state_->target_rotationsq_xyzw, state_->target_scales_2d,
      state_->target_2dgs_mode, state_->target_2dgs_normal_ratio,
      state_->target_2dgs_normal_min, &state_->target_covs);
  } else {
    thrust::device_vector<float> dummy_sq;
    launch_knn(state_.get(),
               state_->target_points, state_->target_points, k_correspondences_,
               &state_->target_knn_idx, &dummy_sq, /*self_knn=*/true,
               knn_backend_);
    launch_covariance(state_->target_points, state_->target_knn_idx, k_correspondences_,
                      regularization_to_int(regularization_method_), &state_->target_covs);
  }
  state_->target_covs_dirty = false;
}

// ---- linearize / compute_error --------------------------------------------
namespace {
PoseT make_pose(const Eigen::Isometry3d& T) {
  PoseT p;
  p.R = T.linear().cast<float>();
  p.t = T.translation().cast<float>();
  return p;
}
}  // namespace

double FastGICPCudaCore::linearize_geometry(
  const Eigen::Isometry3d& trans,
  Eigen::Matrix<double, 6, 6>* H,
  Eigen::Matrix<double, 6, 1>* b) {
  ensure_source_covariances_();
  ensure_target_covariances_();

  const int n_s = static_cast<int>(state_->source_points.size());
  if (n_s == 0) {
    correspondence_residuals_host_.clear();
    if (H) H->setZero();
    if (b) b->setZero();
    return 0.0;
  }

  state_->correspondences.resize(n_s);
  state_->sq_distances.resize(n_s);
  state_->mahalanobis.resize(n_s);
  state_->correspondence_residuals.resize(n_s);

  // Push host-side runtime weights to device if present.
  const bool use_w = !correspondence_weights_host_.empty() &&
                     static_cast<int>(correspondence_weights_host_.size()) == n_s;
  if (use_w) {
    thrust::host_vector<float> w(n_s);
    for (int i = 0; i < n_s; ++i) w[i] = static_cast<float>(correspondence_weights_host_[i]);
    state_->correspondence_weights = w;
  }

  // Output accumulators.
  thrust::device_vector<double> H_dev(36, 0.0);
  thrust::device_vector<double> b_dev(6, 0.0);
  thrust::device_vector<double> cost_dev(1, 0.0);

  PoseT T = make_pose(trans);
  const double max_d = std::isfinite(max_correspondence_distance_) ?
    max_correspondence_distance_ : 1e30;
  const float max_sq = static_cast<float>(max_d * max_d);

  const int block = 128;
  const int grid = (n_s + block - 1) / block;
  linearize_kernel<<<grid, block>>>(
    thrust::raw_pointer_cast(state_->source_points.data()),
    thrust::raw_pointer_cast(state_->target_points.data()),
    thrust::raw_pointer_cast(state_->source_covs.data()),
    thrust::raw_pointer_cast(state_->target_covs.data()),
    n_s, static_cast<int>(state_->target_points.size()),
    T, max_sq,
    use_w ? thrust::raw_pointer_cast(state_->correspondence_weights.data()) : nullptr,
    use_w ? 1 : 0,
    thrust::raw_pointer_cast(state_->correspondences.data()),
    thrust::raw_pointer_cast(state_->sq_distances.data()),
    thrust::raw_pointer_cast(state_->mahalanobis.data()),
    thrust::raw_pointer_cast(state_->correspondence_residuals.data()),
    thrust::raw_pointer_cast(H_dev.data()),
    thrust::raw_pointer_cast(b_dev.data()),
    thrust::raw_pointer_cast(cost_dev.data()));
  FG_CUDA_CHECK(cudaGetLastError());
  FG_CUDA_CHECK(cudaDeviceSynchronize());

  thrust::host_vector<double> H_host = H_dev;
  thrust::host_vector<double> b_host = b_dev;
  thrust::host_vector<double> cost_host = cost_dev;
  if (H) {
    H->setZero();
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) (*H)(r, c) = H_host[r * 6 + c];
  }
  if (b) {
    for (int r = 0; r < 6; ++r) (*b)(r) = b_host[r];
  }

  // Copy residuals back to host.
  thrust::host_vector<float> r_host = state_->correspondence_residuals;
  correspondence_residuals_host_.resize(n_s);
  for (int i = 0; i < n_s; ++i) correspondence_residuals_host_[i] = r_host[i];

  return cost_host[0];
}

double FastGICPCudaCore::compute_error_geometry(const Eigen::Isometry3d& trans) {
  const int n_s = static_cast<int>(state_->source_points.size());
  if (n_s == 0 || state_->correspondences.empty()) return 0.0;
  thrust::device_vector<double> cost_dev(1, 0.0);
  PoseT T = make_pose(trans);
  const bool use_w = !state_->correspondence_weights.empty() &&
                     static_cast<int>(state_->correspondence_weights.size()) == n_s;
  const int block = 128;
  const int grid = (n_s + block - 1) / block;
  cost_kernel<<<grid, block>>>(
    thrust::raw_pointer_cast(state_->source_points.data()),
    thrust::raw_pointer_cast(state_->target_points.data()),
    thrust::raw_pointer_cast(state_->mahalanobis.data()),
    thrust::raw_pointer_cast(state_->correspondences.data()),
    n_s, T,
    use_w ? thrust::raw_pointer_cast(state_->correspondence_weights.data()) : nullptr,
    use_w ? 1 : 0,
    thrust::raw_pointer_cast(cost_dev.data()));
  FG_CUDA_CHECK(cudaGetLastError());
  FG_CUDA_CHECK(cudaDeviceSynchronize());
  thrust::host_vector<double> cost_host = cost_dev;
  return cost_host[0];
}

}  // namespace cuda
}  // namespace fast_gicp
