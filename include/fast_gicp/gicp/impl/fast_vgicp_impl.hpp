#ifndef FAST_GICP_FAST_VGICP_IMPL_HPP
#define FAST_GICP_FAST_VGICP_IMPL_HPP

#include <atomic>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>

#include <fast_gicp/so3/so3.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

namespace fast_gicp {

namespace detail_fast_vgicp {

inline double percentile_from_sorted(const std::vector<double>& values, double q) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  if (values.size() == 1) {
    return values.front();
  }

  const double clamped_q = std::min(1.0, std::max(0.0, q));
  const double scaled = clamped_q * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(scaled));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(scaled));
  if (lower == upper) {
    return values[lower];
  }

  const double t = scaled - static_cast<double>(lower);
  return (1.0 - t) * values[lower] + t * values[upper];
}

}  // namespace detail_fast_vgicp

template <typename PointSource, typename PointTarget>
FastVGICP<PointSource, PointTarget>::FastVGICP() : FastGICP<PointSource, PointTarget>() {
  this->reg_name_ = "FastVGICP";

  voxel_resolution_ = 1.0;
  search_method_ = NeighborSearchMethod::DIRECT1;
  voxel_mode_ = VoxelAccumulationMode::ADDITIVE;
}

template <typename PointSource, typename PointTarget>
FastVGICP<PointSource, PointTarget>::~FastVGICP() {}

template <typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setResolution(double resolution) {
  voxel_resolution_ = resolution;
}

template <typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setNeighborSearchMethod(NeighborSearchMethod method) {
  search_method_ = method;
}

template <typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setVoxelAccumulationMode(VoxelAccumulationMode mode) {
  voxel_mode_ = mode;
}

template <typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  search_source_.swap(search_target_);
  source_covs_.swap(target_covs_);
  voxelmap_.reset();
  voxel_correspondences_.clear();
  voxel_mahalanobis_.clear();
}

template <typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }

  FastGICP<PointSource, PointTarget>::setInputTarget(cloud);
  voxelmap_.reset();
}

template <typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  voxelmap_.reset();

  FastGICP<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  voxel_correspondences_.clear();
  auto offsets = neighbor_offsets(search_method_);
  const bool use_color = this->color_matching_ready();

  std::vector<std::vector<std::pair<int, GaussianVoxel::Ptr>>> corrs(num_threads_);
  for (auto& c : corrs) {
    c.reserve((input_->size() * offsets.size()) / num_threads_);
  }

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    Eigen::Vector4d transed_mean_A = trans * mean_A;
    Eigen::Vector3i coord = voxelmap_->voxel_coord(transed_mean_A);

    if (use_color) {
      GaussianVoxel::Ptr best_voxel = nullptr;
      double best_score = std::numeric_limits<double>::max();
      for (const auto& offset : offsets) {
        auto voxel = voxelmap_->lookup_voxel(coord + offset);
        if (voxel == nullptr) {
          continue;
        }

        const double geometry_score = (voxel->mean - transed_mean_A).template head<3>().squaredNorm();
        const double sigma = std::max(this->color_matching_config_.color_sigma, 1e-9);
        const double color_score =
          this->color_matching_config_.color_weight *
          (this->source_colors_[i] - voxel->mean_color).squaredNorm() /
          (sigma * sigma);
        const double total_score = geometry_score + color_score;
        if (total_score < best_score) {
          best_score = total_score;
          best_voxel = voxel;
        }
      }

      if (best_voxel != nullptr) {
        corrs[omp_get_thread_num()].push_back(std::make_pair(i, best_voxel));
      }
      continue;
    }

    for (const auto& offset : offsets) {
      auto voxel = voxelmap_->lookup_voxel(coord + offset);
      if (voxel != nullptr) {
        corrs[omp_get_thread_num()].push_back(std::make_pair(i, voxel));
      }
    }
  }

  voxel_correspondences_.reserve(input_->size() * offsets.size());
  for (const auto& c : corrs) {
    voxel_correspondences_.insert(voxel_correspondences_.end(), c.begin(), c.end());
  }

  // precompute combined covariances
  voxel_mahalanobis_.resize(voxel_correspondences_.size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < voxel_correspondences_.size(); i++) {
    const auto& corr = voxel_correspondences_[i];
    const auto& cov_A = source_covs_[corr.first];
    const auto& cov_B = corr.second->cov;

    Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
    RCR(3, 3) = 1.0;

    voxel_mahalanobis_[i] = RCR.inverse();
    voxel_mahalanobis_[i](3, 3) = 0.0;
  }
}

template <typename PointSource, typename PointTarget>
double FastVGICP<PointSource, PointTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  if (voxelmap_ == nullptr) {
    voxelmap_.reset(new GaussianVoxelMap<PointTarget>(voxel_resolution_, voxel_mode_));
    if (this->color_matching_ready()) {
      voxelmap_->create_voxelmap(*target_, target_covs_, &this->target_colors_);
    } else {
      voxelmap_->create_voxelmap(*target_, target_covs_);
    }
  }

  update_correspondences(trans);

  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < voxel_correspondences_.size(); i++) {
    const auto& corr = voxel_correspondences_[i];
    auto target_voxel = corr.second;

    const Eigen::Vector4d mean_A = input_->at(corr.first).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[corr.first];

    const Eigen::Vector4d mean_B = corr.second->mean;
    const auto& cov_B = corr.second->cov;

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    double w = std::sqrt(target_voxel->num_points);
    sum_errors += w * error.transpose() * voxel_mahalanobis_[i] * error;

    if (H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = w * jlossexp.transpose() * voxel_mahalanobis_[i] * jlossexp;
    Eigen::Matrix<double, 6, 1> bi = w * jlossexp.transpose() * voxel_mahalanobis_[i] * error;

    int thread_num = omp_get_thread_num();
    Hs[thread_num] += Hi;
    bs[thread_num] += bi;
  }

  if (H && b) {
    H->setZero();
    b->setZero();
    for (int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
double FastVGICP<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans) {
  double sum_errors = 0.0;
#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors)
  for (int i = 0; i < voxel_correspondences_.size(); i++) {
    const auto& corr = voxel_correspondences_[i];
    auto target_voxel = corr.second;

    const Eigen::Vector4d mean_A = input_->at(corr.first).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[corr.first];

    const Eigen::Vector4d mean_B = corr.second->mean;
    const auto& cov_B = corr.second->cov;

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    double w = std::sqrt(target_voxel->num_points);
    sum_errors += w * error.transpose() * voxel_mahalanobis_[i] * error;
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::collect_alignment_quality_metrics(
  AlignmentQualityReport* report,
  const Eigen::Isometry3d& final_pose) const {
  report->used_color_matching = this->color_matching_ready();
  report->has_match_statistics = true;
  report->correspondence_count = static_cast<int>(voxel_correspondences_.size());
  report->matched_count = 0;

  if (input_ == nullptr || input_->empty()) {
    return;
  }

  std::vector<unsigned char> matched(input_->size(), 0);
  std::vector<double> residual_sq_distances;
  residual_sq_distances.reserve(voxel_correspondences_.size());

  for (const auto& corr : voxel_correspondences_) {
    if (corr.first < 0 || corr.first >= input_->size() || corr.second == nullptr) {
      continue;
    }

    matched[corr.first] = 1;

    const Eigen::Vector4d mean_A = input_->at(corr.first).getVector4fMap().template cast<double>();
    const Eigen::Vector4d transformed = final_pose * mean_A;
    const Eigen::Vector3d residual = (corr.second->mean - transformed).template head<3>();
    residual_sq_distances.push_back(residual.squaredNorm());
  }

  report->matched_count = std::accumulate(matched.begin(), matched.end(), 0);
  report->matched_ratio = static_cast<double>(report->matched_count) / static_cast<double>(input_->size());

  if (residual_sq_distances.empty()) {
    return;
  }

  std::sort(residual_sq_distances.begin(), residual_sq_distances.end());
  report->mean_sq_distance =
    std::accumulate(residual_sq_distances.begin(), residual_sq_distances.end(), 0.0) /
    static_cast<double>(residual_sq_distances.size());
  report->median_sq_distance = detail_fast_vgicp::percentile_from_sorted(residual_sq_distances, 0.5);
  report->p90_sq_distance = detail_fast_vgicp::percentile_from_sorted(residual_sq_distances, 0.90);
  report->p95_sq_distance = detail_fast_vgicp::percentile_from_sorted(residual_sq_distances, 0.95);
}

}  // namespace fast_gicp

#endif
