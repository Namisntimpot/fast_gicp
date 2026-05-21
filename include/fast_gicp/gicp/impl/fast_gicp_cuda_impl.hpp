#ifndef FAST_GICP_FAST_GICP_CUDA_IMPL_HPP
#define FAST_GICP_FAST_GICP_CUDA_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <numeric>

#include <fast_gicp/gicp/fast_gicp_cuda.hpp>
#include <fast_gicp/cuda/fast_gicp_cuda_core.cuh>

#include <pcl/common/transforms.h>

namespace fast_gicp {

template <typename PointSource, typename PointTarget>
FastGICPCuda<PointSource, PointTarget>::FastGICPCuda() {
  this->reg_name_ = "FastGICPCuda";
  this->max_iterations_ = 64;
  k_correspondences_ = 20;
  regularization_method_ = RegularizationMethod::PLANE;
  knn_backend_ = "brute_force";
  impl_.reset(new cuda::FastGICPCudaCore());
  impl_->set_correspondence_randomness(k_correspondences_);
  impl_->set_regularization_method(regularization_method_);
  impl_->set_knn_backend(knn_backend_);
}

template <typename PointSource, typename PointTarget>
FastGICPCuda<PointSource, PointTarget>::~FastGICPCuda() = default;

template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
  impl_->set_correspondence_randomness(k);
}

template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
  impl_->set_regularization_method(method);
}

template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::setKnnBackend(const std::string& backend) {
  knn_backend_ = backend;
  impl_->set_knn_backend(backend);
}

template <typename PointSource, typename PointTarget>
const std::string& FastGICPCuda<PointSource, PointTarget>::getKnnBackend() const {
  return knn_backend_;
}

template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  impl_->swap_source_target();
}

template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::clearSource() {
  input_.reset();
  impl_->clear_source();
}

template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::clearTarget() {
  target_.reset();
  impl_->clear_target();
}

namespace detail {

template <typename PointT>
std::vector<float> pcl_to_xyz_flat(const typename pcl::PointCloud<PointT>::ConstPtr& cloud) {
  std::vector<float> out;
  if (!cloud) return out;
  out.resize(static_cast<std::size_t>(cloud->size()) * 3u);
  for (std::size_t i = 0; i < cloud->size(); ++i) {
    out[3 * i + 0] = cloud->at(i).x;
    out[3 * i + 1] = cloud->at(i).y;
    out[3 * i + 2] = cloud->at(i).z;
  }
  return out;
}

}  // namespace detail

template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  auto flat = detail::pcl_to_xyz_flat<PointSource>(cloud);
  impl_->set_source_points(flat.data(), cloud ? static_cast<int>(cloud->size()) : 0);
}

template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  auto flat = detail::pcl_to_xyz_flat<PointTarget>(cloud);
  impl_->set_target_points(flat.data(), cloud ? static_cast<int>(cloud->size()) : 0);
}

template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::setSourceCovariances2DGS(
  const std::vector<float>& rotationsq_xyzw,
  const std::vector<float>& scales_2d,
  const std::string& mode,
  double normal_sigma_ratio,
  double normal_sigma_min) {
  const int n = static_cast<int>(scales_2d.size() / 2);
  if (static_cast<int>(rotationsq_xyzw.size() / 4) != n) {
    throw std::invalid_argument("FastGICPCuda::setSourceCovariances2DGS rotations/scales size mismatch");
  }
  impl_->set_source_covariances_2dgs(rotationsq_xyzw.data(), scales_2d.data(), n,
                                     mode, normal_sigma_ratio, normal_sigma_min);
}

template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::setTargetCovariances2DGS(
  const std::vector<float>& rotationsq_xyzw,
  const std::vector<float>& scales_2d,
  const std::string& mode,
  double normal_sigma_ratio,
  double normal_sigma_min) {
  const int n = static_cast<int>(scales_2d.size() / 2);
  if (static_cast<int>(rotationsq_xyzw.size() / 4) != n) {
    throw std::invalid_argument("FastGICPCuda::setTargetCovariances2DGS rotations/scales size mismatch");
  }
  impl_->set_target_covariances_2dgs(rotationsq_xyzw.data(), scales_2d.data(), n,
                                     mode, normal_sigma_ratio, normal_sigma_min);
}

template <typename PointSource, typename PointTarget>
int FastGICPCuda<PointSource, PointTarget>::getSourceSize() const {
  return impl_->source_size();
}

template <typename PointSource, typename PointTarget>
int FastGICPCuda<PointSource, PointTarget>::getTargetSize() const {
  return impl_->target_size();
}

template <typename PointSource, typename PointTarget>
const std::vector<int>&
FastGICPCuda<PointSource, PointTarget>::getSourceCorrespondences() const {
  impl_->download_correspondences(&correspondences_host_, &sq_distances_host_);
  return correspondences_host_;
}

template <typename PointSource, typename PointTarget>
const std::vector<float>&
FastGICPCuda<PointSource, PointTarget>::getSourceSqDistances() const {
  // download_correspondences fills both buffers; call again here in case the
  // user asked for sq-distances without first asking for indices.
  impl_->download_correspondences(&correspondences_host_, &sq_distances_host_);
  return sq_distances_host_;
}

template <typename PointSource, typename PointTarget>
int FastGICPCuda<PointSource, PointTarget>::current_geometric_term_count() const {
  return impl_->source_size();
}

template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::computeTransformation(
  PointCloudSource& output, const Matrix4& guess) {
  // Hand the latest dynamic-rejection config to the GPU side so it can
  // reproduce the schedule. The base-class computeTransformation drives LM /
  // multi-restart and calls our linearize/compute_error.
  impl_->set_dynamic_rejection_config(this->dynamic_rejection_config_);
  impl_->set_max_correspondence_distance(this->corr_dist_threshold_);
  impl_->reset_iteration_state();
  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget>
double FastGICPCuda<PointSource, PointTarget>::linearize(
  const Eigen::Isometry3d& trans,
  Eigen::Matrix<double, 6, 6>* H,
  Eigen::Matrix<double, 6, 1>* b) {
  // Push the IRLS-computed runtime weights from the base class down to the
  // GPU impl before the kernel runs. (Anchor weights also need refreshing
  // because sparse_anchor_cost runs on CPU via the base class, so its
  // dyn_anchor_weights_ are already consistent — nothing to push for that.)
  impl_->set_correspondence_weights_host(this->dyn_correspondence_weights_);
  const double cost = impl_->linearize_geometry(trans, H, b);
  // Pull the freshly-computed residuals back so the base class can update
  // weights for the next iteration.
  this->dyn_correspondence_residuals_ = impl_->correspondence_residuals_host();
  return cost;
}

template <typename PointSource, typename PointTarget>
double FastGICPCuda<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans) {
  impl_->set_correspondence_weights_host(this->dyn_correspondence_weights_);
  return impl_->compute_error_geometry(trans);
}

// Mirror FastGICP::collect_alignment_quality_metrics so the alignment quality
// report carries non-zero matched_count / correspondence_count / sq-distance
// quantiles. Without this, the base class default (no-op) leaves those at 0,
// which makes downstream code (matched_ratio, normalized_cost_per_match) diverge
// from the CPU path and corrupts keyframe decisions in the SLAM frontend.
template <typename PointSource, typename PointTarget>
void FastGICPCuda<PointSource, PointTarget>::collect_alignment_quality_metrics(
  AlignmentQualityReport* report,
  const Eigen::Isometry3d& final_pose) const {
  (void)final_pose;
  // Force a device->host refresh of the latest correspondence cache.
  impl_->download_correspondences(&correspondences_host_, &sq_distances_host_);

  report->has_match_statistics = true;
  report->correspondence_count = 0;
  report->matched_count = 0;

  std::vector<double> valid_sq_distances;
  valid_sq_distances.reserve(sq_distances_host_.size());
  for (std::size_t i = 0; i < correspondences_host_.size(); ++i) {
    if (correspondences_host_[i] < 0) {
      continue;
    }
    report->matched_count++;
    report->correspondence_count++;
    if (i < sq_distances_host_.size() && std::isfinite(sq_distances_host_[i])) {
      valid_sq_distances.push_back(static_cast<double>(sq_distances_host_[i]));
    }
  }

  if (report->source_count > 0) {
    report->matched_ratio = static_cast<double>(report->matched_count) /
                            static_cast<double>(report->source_count);
  }

  if (valid_sq_distances.empty()) {
    return;
  }
  std::sort(valid_sq_distances.begin(), valid_sq_distances.end());
  const std::size_t n = valid_sq_distances.size();
  report->mean_sq_distance =
    std::accumulate(valid_sq_distances.begin(), valid_sq_distances.end(), 0.0) /
    static_cast<double>(n);
  auto pctl = [&](double p) -> double {
    if (n == 0) return 0.0;
    const double idx = p * static_cast<double>(n - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
    if (lo == hi) return valid_sq_distances[lo];
    const double frac = idx - static_cast<double>(lo);
    return valid_sq_distances[lo] * (1.0 - frac) + valid_sq_distances[hi] * frac;
  };
  report->median_sq_distance = pctl(0.50);
  report->p90_sq_distance = pctl(0.90);
  report->p95_sq_distance = pctl(0.95);
}

}  // namespace fast_gicp

#endif
