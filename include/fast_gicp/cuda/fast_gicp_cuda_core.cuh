#ifndef FAST_GICP_CUDA_FAST_GICP_CUDA_CORE_CUH
#define FAST_GICP_CUDA_FAST_GICP_CUDA_CORE_CUH

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <fast_gicp/gicp/gicp_settings.hpp>
#include <fast_gicp/gicp/dynamic_rejection.hpp>

namespace fast_gicp {
namespace cuda {

// Forward declarations of thrust-typed members are hidden in the .cu via PImpl
// so that consumers can include this header without pulling thrust into the
// translation unit.
struct FastGICPCudaCoreState;

/**
 * @brief Owner of the GPU-side buffers and kernel dispatchers used by
 *        FastGICPCuda. Lives entirely in CUDA-compiled translation units; the
 *        C++ side only sees this opaque handle.
 */
class FastGICPCudaCore {
 public:
  FastGICPCudaCore();
  ~FastGICPCudaCore();

  // Configuration
  void set_correspondence_randomness(int k);
  void set_regularization_method(RegularizationMethod method);
  void set_knn_backend(const std::string& backend);
  const std::string& knn_backend() const { return knn_backend_; }
  void set_max_correspondence_distance(double d);
  void set_num_threads_hint(int n);

  // Data ingestion (Nx3 float arrays in host memory).
  void set_source_points(const float* host_points, int count);
  void set_target_points(const float* host_points, int count);
  void swap_source_target();
  void clear_source();
  void clear_target();
  int source_size() const;
  int target_size() const;

  // 2DGS surfel covariance inputs (rotation quaternion XYZW + 2D scales).
  void set_source_covariances_2dgs(
    const float* rotations_xyzw, const float* scales_2d, int count,
    const std::string& mode, double normal_sigma_ratio, double normal_sigma_min);
  void set_target_covariances_2dgs(
    const float* rotations_xyzw, const float* scales_2d, int count,
    const std::string& mode, double normal_sigma_ratio, double normal_sigma_min);

  // Sparse 3D anchor correspondences (host -> device).
  void set_sparse_anchor_correspondences(
    const double* source_points, const double* target_points, int count,
    const double* weights, int weights_count,
    const double* sigmas, int sigmas_count);
  void clear_sparse_anchor_correspondences();
  void set_use_sparse_anchors(bool enable);

  // Dynamic rejection config + runtime weights.
  void set_dynamic_rejection_config(const DynamicRejectionConfig& cfg);
  const DynamicRejectionConfig& dynamic_rejection_config() const { return dyn_cfg_; }
  const DynamicRejectionDiagnostics& dynamic_rejection_diagnostics() const { return dyn_diag_; }

  // Returns Mahalanobis squared residuals (host vector). Filled by linearize().
  const std::vector<double>& correspondence_residuals_host() const { return correspondence_residuals_host_; }
  const std::vector<double>& correspondence_weights_host() const { return correspondence_weights_host_; }
  // Push host-side weights from the base IRLS code into the GPU impl.
  void set_correspondence_weights_host(const std::vector<double>& w) { correspondence_weights_host_ = w; }
  void set_anchor_weights_host(const std::vector<double>& w) { anchor_weights_host_ = w; }
  const std::vector<double>& anchor_residuals_host() const { return anchor_residuals_host_; }
  const std::vector<double>& anchor_weights_host() const { return anchor_weights_host_; }

  // Update runtime weights from previous iteration's residuals + current mu.
  // Returns nothing; reads dyn_cfg_, writes correspondence_weights_host_ /
  // anchor_weights_host_.
  void prepare_dynamic_weights_for_iteration(int iter);

  // Build correspondences for the given pose then return geometric cost
  // (geometry side only). Optionally fills H and b. Sparse anchor cost is
  // handled separately by the C++ wrapper (it already uses CPU code for the
  // anchor path in the base class).
  double linearize_geometry(
    const Eigen::Isometry3d& trans,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b);

  // Re-evaluate cost only (no correspondence search), using the latest
  // correspondences cached from the most recent linearize_geometry().
  double compute_error_geometry(const Eigen::Isometry3d& trans);

  // Reset iteration-local state at the start of computeTransformation.
  void reset_iteration_state();

  // Download the latest source->target correspondence cache (from the most
  // recent linearize_geometry call) into host memory. `out_corr` is filled
  // with N_s ints (-1 if no correspondence), `out_sq_dist` with N_s squared
  // distances (large sentinel where no correspondence). Used by the frontend
  // for matched_ratio / keyframe_ratio decisions, mirroring CPU FastGICP's
  // getSourceCorrespondences / getSourceSqDistances.
  void download_correspondences(std::vector<int>* out_corr,
                                std::vector<float>* out_sq_dist) const;

 private:
  // Lazy covariance build (runs KNN + per-point cov if dirty).
  void ensure_source_covariances_();
  void ensure_target_covariances_();

  std::unique_ptr<FastGICPCudaCoreState> state_;

  RegularizationMethod regularization_method_ = RegularizationMethod::PLANE;
  int k_correspondences_ = 20;
  std::string knn_backend_ = "brute_force";
  double max_correspondence_distance_ = std::numeric_limits<double>::max();
  int num_threads_hint_ = 0;

  DynamicRejectionConfig dyn_cfg_;
  DynamicRejectionDiagnostics dyn_diag_;
  double gnc_mu_current_ = 0.0;
  double gnc_mu_floor_ = 0.0;

  // Host-side copies of per-iteration data so the C++ wrapper can hand them to
  // the base-class machinery without exposing thrust types.
  std::vector<double> correspondence_residuals_host_;
  std::vector<double> correspondence_weights_host_;
  std::vector<double> anchor_residuals_host_;
  std::vector<double> anchor_weights_host_;
};

}  // namespace cuda
}  // namespace fast_gicp

#endif
