#ifndef FAST_GICP_FAST_GICP_IMPL_HPP
#define FAST_GICP_FAST_GICP_IMPL_HPP

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <numeric>

#include <fast_gicp/so3/so3.hpp>

namespace fast_gicp {

namespace detail_fast_gicp {

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

}  // namespace detail_fast_gicp

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::FastGICP() {
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  k_correspondences_ = 10;  //25
  reg_name_ = "FastGICP";
  corr_dist_threshold_ = std::numeric_limits<float>::max();
  knn_max_distance_ = 0.5;
  
  source_covs_.clear();  
  source_rotationsq_.clear();
  source_scales_.clear();
  source_z_values_.clear();

  target_covs_.clear();
  target_rotationsq_.clear();
  target_scales_.clear();

  regularization_method_ = RegularizationMethod::NORMALIZED_ELLIPSE;
  search_source_.reset(new SearchMethodSource);
  search_target_.reset(new SearchMethodTarget);
  color_matching_config_ = ColorMatchingConfig();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::~FastGICP() {}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setNumThreads(int n) {
  num_threads_ = n;


#ifdef _OPENMP
  if (n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setKNNMaxDistance(float k) {
  knn_max_distance_ = k;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setColorMatchingConfig(const ColorMatchingConfig& config) {
  color_matching_config_ = config;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setColorMatchingEnabled(bool enable) {
  color_matching_config_.enable_color_matching = enable;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceColors(
  const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& colors) {
  source_colors_ = colors;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetColors(
  const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& colors) {
  target_colors_ = colors;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearSourceColors() {
  source_colors_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearTargetColors() {
  target_colors_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
const ColorMatchingConfig& FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::getColorMatchingConfig() const {
  return color_matching_config_;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  search_source_.swap(search_target_);
  source_covs_.swap(target_covs_);
  source_rotationsq_.swap(target_rotationsq_);
  source_scales_.swap(target_scales_);
  source_colors_.swap(target_colors_);

  correspondences_.clear();
  sq_distances_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearSource() {
  input_.reset();
  source_covs_.clear();
  source_rotationsq_.clear();
  source_scales_.clear();
  source_colors_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearTarget() {
  target_.reset();
  target_covs_.clear();
  target_rotationsq_.clear();
  target_scales_.clear();
  target_colors_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  search_source_->setInputCloud(cloud);
  source_covs_.clear();
  source_rotationsq_.clear();
  source_scales_.clear();
  // EXP-124: a new source invalidates any previously-set flow correspondences.
  flow_corr_.clear();
}

// ---- EXP-124: dense-optical-flow correspondences --------------------------------
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setFlowCorrespondences(
    const std::vector<int>& src_idx, const std::vector<int>& tgt_idx) {
  const std::size_t n_src = input_ ? input_->size() : 0;
  flow_corr_.assign(n_src, -1);
  const std::size_t m = std::min(src_idx.size(), tgt_idx.size());
  for (std::size_t k = 0; k < m; k++) {
    const int s = src_idx[k];
    const int t = tgt_idx[k];
    if (s >= 0 && s < (int)n_src && t >= 0) {
      flow_corr_[s] = t;
    }
  }
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setUseFlowCorrespondences(bool use_flow) {
  use_flow_corr_ = use_flow;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearFlowCorrespondences() {
  flow_corr_.clear();
  use_flow_corr_ = false;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateSourceCovariance() {
	if (input_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	source_covs_.clear();
	source_rotationsq_.clear();
	source_scales_.clear();
	calculate_covariances(input_, *search_source_, source_covs_, source_rotationsq_, source_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  if (target_kdtree_mode_ == "incremental") {
    // Dynamic path: rebuild dynamic adaptor instead of the PCL static tree.
    // Bit-identical KNN to a fresh PCL rebuild only modulo tie-break order.
    if (!dynamic_search_target_) {
      dynamic_search_target_.reset(new NanoflannDynamicSearch<PointTarget>());
    }
    dynamic_search_target_->setInputCloud(cloud);
  } else {
    search_target_->setInputCloud(cloud);
  }
  target_covs_.clear();
  target_rotationsq_.clear();
  target_scales_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetKdtreeMode(const std::string& mode) {
  if (mode != "static" && mode != "incremental") {
    throw std::invalid_argument("FastGICP::setTargetKdtreeMode: unknown mode '" + mode + "' (expected 'static' or 'incremental')");
  }
  target_kdtree_mode_ = mode;
  if (mode == "incremental" && !dynamic_search_target_) {
    dynamic_search_target_.reset(new NanoflannDynamicSearch<PointTarget>());
  }
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
int FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::appendInputTarget(
    const std::vector<float>& xyz_flat,
    const std::vector<float>& rotationsq_xyzw,
    const std::vector<float>& scales_2d,
    const std::string& mode,
    double normal_sigma_ratio,
    double normal_sigma_min) {
  if (target_kdtree_mode_ != "incremental") {
    throw std::runtime_error("FastGICP::appendInputTarget requires target_kdtree_mode='incremental'");
  }
  if (!dynamic_search_target_) {
    dynamic_search_target_.reset(new NanoflannDynamicSearch<PointTarget>());
  }
  const std::size_t n_new = xyz_flat.size() / 3;
  if (xyz_flat.size() != 3 * n_new) {
    throw std::invalid_argument("appendInputTarget: xyz_flat size must be multiple of 3");
  }
  if (rotationsq_xyzw.size() != 4 * n_new || scales_2d.size() != 2 * n_new) {
    throw std::invalid_argument("appendInputTarget: rot/scale size mismatch with n_new");
  }
  const std::size_t first_idx = dynamic_search_target_->appendPoints(xyz_flat.data(), n_new);

  // Sync the PCL target cloud with appended points so target_->at(idx) still
  // maps to the correct xyz inside update_correspondences and linearize. The
  // base-class member target_ is a ConstPtr; we keep a mutable shadow we own.
  PointCloudTargetPtr mut_cloud;
  if (target_) {
    // Cast away const on the cloud we previously installed; we own the
    // storage in incremental mode, so this is safe.
    mut_cloud = std::const_pointer_cast<PointCloudTarget>(target_);
  } else {
    mut_cloud.reset(new PointCloudTarget());
  }
  mut_cloud->points.reserve(mut_cloud->size() + n_new);
  for (std::size_t i = 0; i < n_new; ++i) {
    PointTarget p;
    p.x = xyz_flat[3 * i + 0];
    p.y = xyz_flat[3 * i + 1];
    p.z = xyz_flat[3 * i + 2];
    mut_cloud->points.push_back(p);
  }
  mut_cloud->width = static_cast<std::uint32_t>(mut_cloud->points.size());
  mut_cloud->height = 1;
  // Re-install via base-class setInputTarget so target_ reflects the growth.
  // We bypass our override because that would clear covs/rot/scales.
  pcl::Registration<PointSource, PointTarget, Scalar>::target_ = mut_cloud;

  // Extend per-point 2DGS attributes and compute covariances for the new
  // slice only. Uses the same regularization path as setTargetCovariances2DGS.
  const std::size_t old_n = target_covs_.size();
  target_covs_.resize(old_n + n_new);
  target_rotationsq_.resize(4 * (old_n + n_new));
  target_scales_.resize(3 * (old_n + n_new));

  if (mode != "physical" && mode != "normalized") {
    throw std::invalid_argument("appendInputTarget: unknown 2DGS mode '" + mode + "'");
  }

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < static_cast<int>(n_new); ++i) {
    const std::size_t k = old_n + i;
    const double s1 = std::max(static_cast<double>(scales_2d[2 * i + 0]), 1e-12);
    const double s2 = std::max(static_cast<double>(scales_2d[2 * i + 1]), 1e-12);
    const double sn = std::max(normal_sigma_ratio * std::min(s1, s2), normal_sigma_min);

    target_scales_[3 * k + 0] = static_cast<float>(s1);
    target_scales_[3 * k + 1] = static_cast<float>(s2);
    target_scales_[3 * k + 2] = static_cast<float>(sn);

    const double qx = static_cast<double>(rotationsq_xyzw[4 * i + 0]);
    const double qy = static_cast<double>(rotationsq_xyzw[4 * i + 1]);
    const double qz = static_cast<double>(rotationsq_xyzw[4 * i + 2]);
    const double qw = static_cast<double>(rotationsq_xyzw[4 * i + 3]);
    target_rotationsq_[4 * k + 0] = static_cast<float>(qx);
    target_rotationsq_[4 * k + 1] = static_cast<float>(qy);
    target_rotationsq_[4 * k + 2] = static_cast<float>(qz);
    target_rotationsq_[4 * k + 3] = static_cast<float>(qw);
    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();

    Eigen::Vector3d singular_values;
    if (mode == "normalized") {
      singular_values = Eigen::Vector3d(1.0, 1.0, 1e-3);
    } else {
      singular_values = Eigen::Vector3d(s1 * s1, s2 * s2, sn * sn);
    }
    target_covs_[k].setZero();
    target_covs_[k].template block<3, 3>(0, 0) =
        q.toRotationMatrix() * singular_values.asDiagonal() * q.toRotationMatrix().transpose();
  }
  return static_cast<int>(first_idx);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
int FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::removeFromInputTarget(
    const std::vector<int>& indices) {
  if (target_kdtree_mode_ != "incremental") {
    throw std::runtime_error("FastGICP::removeFromInputTarget requires target_kdtree_mode='incremental'");
  }
  if (!dynamic_search_target_) return 0;
  return static_cast<int>(dynamic_search_target_->removePoints(
      reinterpret_cast<const std::int32_t*>(indices.data()), indices.size()));
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
std::vector<int> FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::rebuildTargetKdtree() {
  if (target_kdtree_mode_ != "incremental") {
    throw std::runtime_error("FastGICP::rebuildTargetKdtree requires target_kdtree_mode='incremental'");
  }
  if (!dynamic_search_target_) return {};
  std::vector<std::int32_t> remap = dynamic_search_target_->compact();

  // Re-index target_ + 2DGS arrays to match the compaction.
  const std::size_t old_total = remap.size();
  std::size_t live = 0;
  for (auto v : remap) {
    if (v >= 0) ++live;
  }
  PointCloudTargetPtr new_target(new PointCloudTarget());
  new_target->points.resize(live);
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> new_covs(live);
  std::vector<float> new_rot(4 * live);
  std::vector<float> new_scales(3 * live);
  for (std::size_t i = 0; i < old_total; ++i) {
    const std::int32_t j = remap[i];
    if (j < 0) continue;
    new_target->points[j] = target_->at(i);
    new_covs[j] = target_covs_[i];
    for (int d = 0; d < 4; ++d) new_rot[4 * j + d] = target_rotationsq_[4 * i + d];
    for (int d = 0; d < 3; ++d) new_scales[3 * j + d] = target_scales_[3 * i + d];
  }
  new_target->width = static_cast<std::uint32_t>(live);
  new_target->height = 1;
  target_ = new_target;
  pcl::Registration<PointSource, PointTarget, Scalar>::target_ = target_;
  target_covs_.swap(new_covs);
  target_rotationsq_.swap(new_rot);
  target_scales_.swap(new_scales);
  return std::vector<int>(remap.begin(), remap.end());
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
std::size_t FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::getLiveTargetCount() const {
  return dynamic_search_target_ ? dynamic_search_target_->liveCount() : (target_ ? target_->size() : 0);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
std::size_t FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::getTotalTargetCount() const {
  return dynamic_search_target_ ? dynamic_search_target_->totalCount() : (target_ ? target_->size() : 0);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::getTargetTombstoneRatio() const {
  if (!dynamic_search_target_) return 0.0;
  const std::size_t total = dynamic_search_target_->totalCount();
  if (total == 0) return 0.0;
  return static_cast<double>(dynamic_search_target_->tombstonedCount()) / static_cast<double>(total);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::queryTargetNN(
    const std::vector<float>& query_xyz, std::vector<int>& indices, std::vector<float>& sq_distances) {
  const int n = static_cast<int>(query_xyz.size() / 3);
  indices.assign(n, -1);
  sq_distances.assign(n, std::numeric_limits<float>::infinity());
  const bool use_dynamic = (target_kdtree_mode_ == "incremental") && (dynamic_search_target_ != nullptr);
  if (!use_dynamic && !search_target_) return;
  // CANONICAL_TIEBREAK (default OFF): match update_correspondences so the flow query resolves the
  // SAME physical target point in nanoflann (incremental) and PCL (static) at near-ties.
  static const bool kCanonicalTie = []() { const char* e = std::getenv("CANONICAL_TIEBREAK"); return e && std::atoi(e) != 0; }();
  static const int kTieK = []() { const char* e = std::getenv("CANONICAL_TIE_K"); return e ? std::max(2, std::atoi(e)) : 4; }();
  const int kq = kCanonicalTie ? kTieK : 1;
#pragma omp parallel for num_threads(num_threads_) schedule(guided, 64)
  for (int i = 0; i < n; i++) {
    PointTarget pt;
    pt.x = query_xyz[3 * i + 0];
    pt.y = query_xyz[3 * i + 1];
    pt.z = query_xyz[3 * i + 2];
    std::vector<int> k_idx(kq);
    std::vector<float> k_d2(kq);
    int found = use_dynamic ? dynamic_search_target_->nearestKSearch(pt, kq, k_idx, k_d2)
                            : search_target_->nearestKSearch(pt, kq, k_idx, k_d2);
    if (found > 0) {
      int best = k_idx[0]; float best_d2 = k_d2[0];
      if (kCanonicalTie && best >= 0) {
        const Eigen::Vector3d q(static_cast<double>(pt.x), static_cast<double>(pt.y), static_cast<double>(pt.z));
        double best_d2d = std::numeric_limits<double>::max();
        float bx = 0.f, by = 0.f, bz = 0.f; best = -1;
        for (int c = 0; c < found; c++) {
          const int idx = k_idx[c]; if (idx < 0) continue;
          const auto& p = target_->at(idx);
          const Eigen::Vector3d tp(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z));
          const double d2d = (q - tp).squaredNorm();
          bool take = d2d < best_d2d;
          if (!take && d2d == best_d2d) take = (p.x < bx || (p.x == bx && (p.y < by || (p.y == by && p.z < bz))));
          if (take) { best_d2d = d2d; best = idx; best_d2 = k_d2[c]; bx = p.x; by = p.y; bz = p.z; }
        }
      }
      indices[i] = best;
      sq_distances[i] = best_d2;
    }
  }
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::color_matching_ready() const {
  return color_matching_config_.enable_color_matching &&
         color_matching_config_.color_weight > 0.0 &&
         source_colors_.size() == input_->size() &&
         target_colors_.size() == target_->size();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::color_distance_score(int source_index, int target_index) const {
  if (!color_matching_ready()) {
    return 0.0;
  }

  const double sigma = std::max(color_matching_config_.color_sigma, 1e-9);
  const Eigen::Vector3d diff = source_colors_[source_index] - target_colors_[target_index];
  return color_matching_config_.color_weight * diff.squaredNorm() / (sigma * sigma);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateTargetCovariance() {
	if (target_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	target_covs_.clear();
	target_rotationsq_.clear();
	target_scales_.clear();
	calculate_covariances(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateTargetCovarianceWithZ() {
	if (target_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	target_covs_.clear();
	target_rotationsq_.clear();
	target_scales_.clear();
	calculate_covariances_withz(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_, target_z_values_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateTargetCovarianceWithFilter() {
	if (target_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	target_covs_.clear();
	target_rotationsq_.clear();
	target_scales_.clear();
	calculate_target_covariances_with_filter(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_, target_filter_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  source_covs_ = covs;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales)
	{
		setCovariances(input_rotationsq, input_scales, source_covs_, source_rotationsq_, source_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceCovariances2DGS(
	const std::vector<float>& input_rotationsq_xyzw,
	const std::vector<float>& input_scales_2d,
	const std::string& mode,
	double normal_sigma_ratio,
	double normal_sigma_min)
	{
		setCovariances2DGS(input_rotationsq_xyzw, input_scales_2d, mode, normal_sigma_ratio, normal_sigma_min, source_covs_, source_rotationsq_, source_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceZvalues(const std::vector<float>& input_z_values)
	{
		source_z_values_.clear();
    source_z_values_ = input_z_values;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceFilter(const int num_trackable_points, const std::vector<int>& input_filter)
	{
    source_num_trackable_points_ = num_trackable_points;
		source_filter_.clear();
    source_filter_ = input_filter;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetFilter(const int num_trackable_points, const std::vector<int>& input_filter)
	{
    target_num_trackable_points_ = num_trackable_points;
		target_filter_.clear();
    target_filter_ = input_filter;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetZvalues(const std::vector<float>& input_z_values)
	{
		target_z_values_.clear();
    target_z_values_ = input_z_values;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales)
	{
		setCovariances(input_rotationsq, input_scales, target_covs_, target_rotationsq_, target_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetCovariances2DGS(
	const std::vector<float>& input_rotationsq_xyzw,
	const std::vector<float>& input_scales_2d,
	const std::string& mode,
	double normal_sigma_ratio,
	double normal_sigma_min)
	{
		setCovariances2DGS(input_rotationsq_xyzw, input_scales_2d, mode, normal_sigma_ratio, normal_sigma_min, target_covs_, target_rotationsq_, target_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  if (output.points.data() == input_->points.data() || output.points.data() == target_->points.data()) {
    throw std::invalid_argument("FastGICP: destination cloud cannot be identical to source or target");
  }
  if (source_covs_.size() != input_->size()) {
    if (!source_filter_.empty() && source_filter_.size() == input_->size() && source_num_trackable_points_ > 0) {
      calculate_source_covariances_with_filter(input_, *search_source_, source_covs_, source_rotationsq_, source_scales_, source_filter_);
    } else {
      calculate_covariances(input_, *search_source_, source_covs_, source_rotationsq_, source_scales_);
    }
  }
  if (target_covs_.size() != target_->size()) {
    if (!target_filter_.empty() && target_filter_.size() == target_->size() && target_num_trackable_points_ > 0) {
      calculate_target_covariances_with_filter(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_, target_filter_);
    } else {
      calculate_covariances(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_);
    }
  }
  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_.size() == input_->size());
  assert(target_covs_.size() == target_->size());

  Eigen::Isometry3f trans_f = trans.cast<float>();

  correspondences_.resize(input_->size());
  sq_distances_.resize(input_->size());
  mahalanobis_.resize(input_->size());

  const bool use_color = color_matching_ready();
  // CANONICAL_TIEBREAK (env, default OFF -> byte-identical): two EXACT NN methods (nanoflann vs
  // PCL FLANN) can only disagree on near-ties (target points within float-eps of equidistant from
  // a query); they then pick different equidistant points -> chaotic divergence on long seqs (the
  // INCR_KD-vs-PCL 688mm). Fix: fetch the K nearest (both libs return the SAME K) and, among
  // candidates whose sq_dist is within tie_eps of the minimum, pick the one with the
  // lexicographically smallest target point (x,y,z) -- index/library-agnostic -> identical choice
  // in both the incremental (nanoflann) and static (PCL) trees. Cost ~0 off near-ties.
  static const bool kCanonicalTie = []() { const char* e = std::getenv("CANONICAL_TIEBREAK"); return e && std::atoi(e) != 0; }();
  static const int kTieK = []() { const char* e = std::getenv("CANONICAL_TIE_K"); return e ? std::max(2, std::atoi(e)) : 4; }();
  const int candidate_count = use_color ? std::max(1, color_matching_config_.geometric_candidate_count)
                                        : (kCanonicalTie ? kTieK : 1);
  std::vector<int> k_indices(candidate_count);
  std::vector<float> k_sq_dists(candidate_count);

  const bool use_dynamic = (target_kdtree_mode_ == "incremental") && (dynamic_search_target_ != nullptr);

#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    PointTarget pt;

    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();

    // EXP-124: dense-flow correspondence OVERRIDE. If a flow target index is set
    // for this source point, use it directly (bypassing the NN kdtree search AND
    // the corr_dist_threshold_ gate), build the Mahalanobis matrix from that
    // target's covariance (identical RCR formula to the NN path), and skip ahead.
    if (use_flow_corr_ && i < (int)flow_corr_.size() && flow_corr_[i] >= 0) {
      const int target_index = flow_corr_[i];
      // squared residual between transformed source and the flow-matched target
      const Eigen::Vector4f resid = pt.getVector4fMap() - target_->at(target_index).getVector4fMap();
      sq_distances_[i] = resid.head<3>().squaredNorm();
      correspondences_[i] = target_index;

      const auto& cov_A = source_covs_[i];
      const auto& cov_B = target_covs_[target_index];
      Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
      RCR(3, 3) = 1.0;
      if (RCR.determinant() == 0) {
        mahalanobis_[i] = RCR.completeOrthogonalDecomposition().pseudoInverse();
        mahalanobis_[i](3, 3) = 0.0f;
      } else {
        mahalanobis_[i] = RCR.inverse();
        mahalanobis_[i](3, 3) = 0.0f;
      }
      continue;
    }

    if (use_dynamic) {
      dynamic_search_target_->nearestKSearch(pt, candidate_count, k_indices, k_sq_dists);
    } else {
      search_target_->nearestKSearch(pt, candidate_count, k_indices, k_sq_dists);
    }

    int best_index = -1;
    float best_sq_dist = std::numeric_limits<float>::max();
    double best_score = std::numeric_limits<double>::max();
    for (int candidate = 0; candidate < k_indices.size(); candidate++) {
      if (k_sq_dists[candidate] >= corr_dist_threshold_ * corr_dist_threshold_) {
        continue;
      }

      const double score = static_cast<double>(k_sq_dists[candidate]) + color_distance_score(i, k_indices[candidate]);
      if (score < best_score) {
        best_score = score;
        best_sq_dist = k_sq_dists[candidate];
        best_index = k_indices[candidate];
      }
    }

    // CANONICAL tie-break (geometry only): the nanoflann(incremental) vs PCL(static) disagreement
    // is pure float32 rounding in the per-candidate squared-distance. RE-RANK the K candidates by
    // a float64 recompute of ||pt - target|| from the SAME query pt and SAME target point -> both
    // libraries compute the identical double value and pick the identical true-nearest. This is
    // accuracy-OPTIMAL (picks the genuine NN, no lexicographic spatial bias) and deterministic.
    // Exact float64 ties (astronomically rare) fall back to the lexicographically smallest point.
    if (kCanonicalTie && !use_color && best_index >= 0) {
      const Eigen::Vector3d q(static_cast<double>(pt.x), static_cast<double>(pt.y), static_cast<double>(pt.z));
      double best_d2d = std::numeric_limits<double>::max();
      const float gate2 = corr_dist_threshold_ * corr_dist_threshold_;
      float bx = 0.f, by = 0.f, bz = 0.f;
      for (int candidate = 0; candidate < (int)k_indices.size(); candidate++) {
        if (k_sq_dists[candidate] >= gate2) continue;
        const int idx = k_indices[candidate];
        if (idx < 0) continue;
        const auto& p = target_->at(idx);
        const Eigen::Vector3d tp(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z));
        const double d2d = (q - tp).squaredNorm();
        bool take = d2d < best_d2d;
        if (!take && d2d == best_d2d) {  // exact-tie canonical fallback
          take = (p.x < bx || (p.x == bx && (p.y < by || (p.y == by && p.z < bz))));
        }
        if (take) { best_d2d = d2d; best_index = idx; best_sq_dist = k_sq_dists[candidate]; bx = p.x; by = p.y; bz = p.z; }
      }
    }

    sq_distances_[i] = best_sq_dist;
    correspondences_[i] = best_index;
    if (correspondences_[i] < 0) {
      continue;
    }

    const int target_index = correspondences_[i];
    const auto& cov_A = source_covs_[i];
    const auto& cov_B = target_covs_[target_index];

    Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
    RCR(3, 3) = 1.0;

    if (RCR.determinant() == 0){
      // std::cout << "mahalanobis value will be NaN" << std::endl;
      mahalanobis_[i] = RCR.completeOrthogonalDecomposition().pseudoInverse();
      mahalanobis_[i](3, 3) = 0.0f;
    }
    else{
      mahalanobis_[i] = RCR.inverse();
      mahalanobis_[i](3, 3) = 0.0f;
    }
  }
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::linearize(
  const Eigen::Isometry3d& trans,
  typename LsqRegistration<PointSource, PointTarget>::Matrix6* H,
  typename LsqRegistration<PointSource, PointTarget>::Vector6* b) {
  update_correspondences(trans);

  // Resize residual buffer in the base class (sentinel -1 == no correspondence)
  this->dyn_correspondence_residuals_.assign(input_->size(), -1.0);
  const bool use_dyn_weights = !this->dyn_correspondence_weights_.empty() &&
                               this->dyn_correspondence_weights_.size() == input_->size();

  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    const double r2 = error.transpose() * mahalanobis_[i] * error;
    this->dyn_correspondence_residuals_[i] = r2;

    const double w = use_dyn_weights ? std::max(0.0, std::min(1.0, this->dyn_correspondence_weights_[i])) : 1.0;

    sum_errors += w * r2;

    if (H == nullptr || b == nullptr || w == 0.0) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = w * (jlossexp.transpose() * mahalanobis_[i] * jlossexp);
    Eigen::Matrix<double, 6, 1> bi = w * (jlossexp.transpose() * mahalanobis_[i] * error);

    Hs[omp_get_thread_num()] += Hi;
    bs[omp_get_thread_num()] += bi;
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

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::compute_error(const Eigen::Isometry3d& trans) {
  double sum_errors = 0.0;
  const bool use_dyn_weights = !this->dyn_correspondence_weights_.empty() &&
                               this->dyn_correspondence_weights_.size() == input_->size();

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    const double w = use_dyn_weights ? std::max(0.0, std::min(1.0, this->dyn_correspondence_weights_[i])) : 1.0;
    const double r2 = error.transpose() * mahalanobis_[i] * error;
    sum_errors += w * r2;
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
int FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::current_geometric_term_count() const {
  return static_cast<int>(std::count_if(
    correspondences_.begin(),
    correspondences_.end(),
    [](int index) { return index >= 0; }));
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::collect_alignment_quality_metrics(
  AlignmentQualityReport* report,
  const Eigen::Isometry3d& final_pose) const {
  (void)final_pose;

  report->used_color_matching = color_matching_ready();
  report->has_match_statistics = true;
  report->correspondence_count = 0;
  report->matched_count = 0;

  std::vector<double> valid_sq_distances;
  valid_sq_distances.reserve(sq_distances_.size());
  for (std::size_t i = 0; i < correspondences_.size(); i++) {
    if (correspondences_[i] < 0) {
      continue;
    }

    report->matched_count++;
    report->correspondence_count++;
    if (std::isfinite(sq_distances_[i])) {
      valid_sq_distances.push_back(static_cast<double>(sq_distances_[i]));
    }
  }

  if (report->source_count > 0) {
    report->matched_ratio = static_cast<double>(report->matched_count) / static_cast<double>(report->source_count);
  }

  if (valid_sq_distances.empty()) {
    return;
  }

  std::sort(valid_sq_distances.begin(), valid_sq_distances.end());
  report->mean_sq_distance =
    std::accumulate(valid_sq_distances.begin(), valid_sq_distances.end(), 0.0) / static_cast<double>(valid_sq_distances.size());
  report->median_sq_distance = detail_fast_gicp::percentile_from_sorted(valid_sq_distances, 0.5);
  report->p90_sq_distance = detail_fast_gicp::percentile_from_sorted(valid_sq_distances, 0.90);
  report->p95_sq_distance = detail_fast_gicp::percentile_from_sorted(valid_sq_distances, 0.95);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_covariances(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  covariances.resize(cloud->size());
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }

    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
      }
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    
    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());
    qfrommat.normalize();
//    Eigen::Vector4d q = {qfrommat.x(), qfrommat.y(), qfrommat.z(), qfrommat.w()};
    // rotationsq.insert(rotationsq.end(), { (float)qfrommat.x(), (float)qfrommat.y(), (float)qfrommat.z(), (float)qfrommat.w()});
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();
    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();
    // scales.insert(scales.end(), {(float)scale.x(), (float)scale.y(), (float)scale.z()});
    scales[3*i+0] = (float)scale.x();
    scales[3*i+1] = (float)scale.y();
    scales[3*i+2] = (float)scale.z();

    // compute regularized covariance
    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;
      switch (regularization_method_) {
        default:
          std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_ELLIPSE:
          // std::cout<<svd.singularValues()(1)<<std::endl;
          if (svd.singularValues()(1) == 0){
          	values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
          }
          else{          
            values = svd.singularValues() / svd.singularValues()(1);
            values = values.array().max(1e-3);
	        }
      }
      // use regularized covariance
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  }

  return true;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_covariances_withz(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales,
  std::vector<float>& z_values
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  covariances.resize(cloud->size());
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }

    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
      }
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    
    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());
    qfrommat.normalize();
//    Eigen::Vector4d q = {qfrommat.x(), qfrommat.y(), qfrommat.z(), qfrommat.w()};
    // rotationsq.insert(rotationsq.end(), { (float)qfrommat.x(), (float)qfrommat.y(), (float)qfrommat.z(), (float)qfrommat.w()});
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();
    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();
    // scales.insert(scales.end(), {(float)scale.x(), (float)scale.y(), (float)scale.z()});
    float z = std::max(1.,pow(z_values[i], 1.5)*2.);
    // std::cout<<z<<std::endl;
    scales[3*i+0] = (float)scale.x()/z;
    scales[3*i+1] = (float)scale.y()/z;
    scales[3*i+2] = (float)scale.z()/z;

    // compute regularized covariance
    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;
      switch (regularization_method_) {
        default:
          std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_ELLIPSE:
          // std::cout<<svd.singularValues()(1)<<std::endl;
          if (svd.singularValues()(1) == 0){
          	values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
          }
          else{          
            values = svd.singularValues() / svd.singularValues()(1);
            values = values.array().max(1e-3);
	        }
          break;

      }
      // use regularized covariance
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  }
  return true;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_source_covariances_with_filter(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales,
  std::vector<int>& filter
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  

  typename pcl::PointCloud<PointT>::Ptr newCloud(new pcl::PointCloud<PointT>);
  newCloud->points.resize(source_num_trackable_points_);
  // pcl::copyPointCloud(*cloud, filter, *newCloud);

  // save covariances of trackable points
  covariances.resize(source_num_trackable_points_);
  // calculate and save rot/scales about all points
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);
    
    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }


    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
      }
    }

    
    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    
    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());
    qfrommat.normalize();
//    Eigen::Vector4d q = {qfrommat.x(), qfrommat.y(), qfrommat.z(), qfrommat.w()};
    // rotationsq.insert(rotationsq.end(), { (float)qfrommat.x(), (float)qfrommat.y(), (float)qfrommat.z(), (float)qfrommat.w()});
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();
    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();
    // scales.insert(scales.end(), {(float)scale.x(), (float)scale.y(), (float)scale.z()});
    scales[3*i+0] = (float)scale.x();
    scales[3*i+1] = (float)scale.y();
    scales[3*i+2] = (float)scale.z();

    // Save covariance and xyz of trackable points
    if (filter[i]!=0){
      // compute regularized covariance
      if (regularization_method_ == RegularizationMethod::NONE) {
        covariances[filter[i]-1] = cov;
      } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
        double lambda = 1e-3;
        Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d C_inv = C.inverse();
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
      } else {
        // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d values;
        switch (regularization_method_) {
          default:
            std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
            abort();
          case RegularizationMethod::PLANE:
            values = Eigen::Vector3d(1e-3, 1e-3, 1e-5); //1,1,1e-3
            break;
          case RegularizationMethod::MIN_EIG:
            values = svd.singularValues().array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_MIN_EIG:
            values = svd.singularValues() / svd.singularValues().maxCoeff();
            values = values.array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_ELLIPSE:
            // std::cout<<svd.singularValues()(1)<<std::endl;
            if (svd.singularValues()(1) == 0){
              values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
            }
            else{          
              values = svd.singularValues() / svd.singularValues()(1);
              // values = values.array().max(1e-3);
            }
            break;
          case RegularizationMethod::TEST:
            values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); //1,1,1e-3
        }
        // use regularized covariance
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
        newCloud->points[filter[i]-1] = cloud->at(i);
      }
    }
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(newCloud);
  
  search_source_->setInputCloud(newCloud);
  // std::cout << "Cloud size : " << newCloud->size() << "/cov size : " << covariances.size() << "/rots size : " << rotationsq.size()/4 << std::endl;
  return true;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_target_covariances_with_filter(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales,
  std::vector<int>& filter
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }

  typename pcl::PointCloud<PointT>::Ptr newCloud(new pcl::PointCloud<PointT>);
  newCloud->points.resize(target_num_trackable_points_);
  // pcl::copyPointCloud(*cloud, filter, *newCloud);

  // save covariances of trackable points
  covariances.resize(target_num_trackable_points_);
  // calculate and save rot/scales about all points
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }

    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
      }
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    
    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());
    qfrommat.normalize();
//    Eigen::Vector4d q = {qfrommat.x(), qfrommat.y(), qfrommat.z(), qfrommat.w()};
    // rotationsq.insert(rotationsq.end(), { (float)qfrommat.x(), (float)qfrommat.y(), (float)qfrommat.z(), (float)qfrommat.w()});
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();
    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();
    // scales.insert(scales.end(), {(float)scale.x(), (float)scale.y(), (float)scale.z()});
    scales[3*i+0] = (float)scale.x();
    scales[3*i+1] = (float)scale.y();
    scales[3*i+2] = (float)scale.z();

    // Save covariances of trackable points
    if (filter[i]!=0){
      // compute regularized covariance
      if (regularization_method_ == RegularizationMethod::NONE) {
        covariances[filter[i]-1] = cov;
      } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
        double lambda = 1e-3;
        Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d C_inv = C.inverse();
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
      } else {
        // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d values;
        switch (regularization_method_) {
          default:
            std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
            abort();
          case RegularizationMethod::PLANE:
            values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); //1,1,1e-3
            break;
          case RegularizationMethod::MIN_EIG:
            values = svd.singularValues().array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_MIN_EIG:
            values = svd.singularValues() / svd.singularValues().maxCoeff();
            values = values.array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_ELLIPSE:
            // std::cout<<svd.singularValues()(1)<<std::endl;
            if (svd.singularValues()(1) == 0){
              values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
            }
            else{          
              values = svd.singularValues() / svd.singularValues()(1);
              // values = values.array().max(1e-3);
            }
            break;
          case RegularizationMethod::TEST:
            values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); //1,1,1e-3
        }
        // use regularized covariance
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
        newCloud->points[filter[i]-1] = cloud->at(i);
      }
    }
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(newCloud);
  search_target_->setInputCloud(newCloud);
  // std::cout << "Cloud size : " << newCloud->size() << "/cov size : " << covariances.size() << "/rots size : " << rotationsq.size()/4 << std::endl;
  // std::cout << "Checker : " << checker << std::endl;
  return true;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setCovariances2DGS(
	const std::vector<float>& input_rotationsq_xyzw,
	const std::vector<float>& input_scales_2d,
	const std::string& mode,
	double normal_sigma_ratio,
	double normal_sigma_min,
	std::vector<Eigen::Matrix4d,
  Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales)
	{
	if(input_rotationsq_xyzw.size()/4 != input_scales_2d.size()/2){
		throw std::invalid_argument("2DGS rotation/scale size mismatch");
	}
	if(mode != "physical" && mode != "normalized"){
		throw std::invalid_argument("unknown 2DGS covariance mode: " + mode);
	}
	if(normal_sigma_ratio < 0.0 || normal_sigma_min < 0.0){
		throw std::invalid_argument("2DGS normal sigma parameters must be non-negative");
	}

	const std::size_t n = input_scales_2d.size()/2;
	rotationsq.clear();
	scales.clear();
	rotationsq = input_rotationsq_xyzw;
	scales.resize(3*n);
	covariances.resize(n);

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
	for(int i=0; i<static_cast<int>(n); i++){
		const double s1 = std::max(static_cast<double>(input_scales_2d[2*i+0]), 1e-12);
		const double s2 = std::max(static_cast<double>(input_scales_2d[2*i+1]), 1e-12);
		const double sn = std::max(normal_sigma_ratio * std::min(s1, s2), normal_sigma_min);

		scales[3*i+0] = static_cast<float>(s1);
		scales[3*i+1] = static_cast<float>(s2);
		scales[3*i+2] = static_cast<float>(sn);

		const double x = static_cast<double>(input_rotationsq_xyzw[4*i+0]);
		const double y = static_cast<double>(input_rotationsq_xyzw[4*i+1]);
		const double z = static_cast<double>(input_rotationsq_xyzw[4*i+2]);
		const double w = static_cast<double>(input_rotationsq_xyzw[4*i+3]);
		Eigen::Quaterniond q(w, x, y, z);
		q.normalize();

		Eigen::Vector3d singular_values;
		if(mode == "normalized"){
			singular_values = Eigen::Vector3d(1.0, 1.0, 1e-3);
		} else {
			singular_values = Eigen::Vector3d(s1*s1, s2*s2, sn*sn);
		}

		covariances[i].setZero();
		covariances[i].template block<3, 3>(0, 0) = q.toRotationMatrix() * singular_values.asDiagonal() * q.toRotationMatrix().transpose();
	}
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales,
	std::vector<Eigen::Matrix4d,
  Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales) 
	{
	if(input_rotationsq.size()/4 != input_scales.size()/3){
		std::cerr << "size not match" <<std::endl;
		abort();
	}
	rotationsq.clear();
	scales.clear();
	rotationsq = input_rotationsq;
	scales = input_scales;
  // covariances.resize(input_scales.size());
	covariances.resize(input_scales.size()/3);
	// rotationsq.resize(input_scales.size());
	// scales.resize(input_scales.size());
  // clock_t start_time = clock();
#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
	for(int i=0; i<scales.size()/3; i++){
		Eigen::Vector3d singular_values = { (double)scales[3*i+0]*scales[3*i+0], 
							(double)scales[3*i+1]*scales[3*i+1], 
							(double)scales[3*i+2]*scales[3*i+2] };
		switch (regularization_method_) {
		default:
		  std::cerr << "here must not be reached" << std::endl;
		  abort();
		case RegularizationMethod::PLANE:
		  singular_values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); //1,1,1e-3
		  break;
		case RegularizationMethod::MIN_EIG:
		  singular_values = singular_values.array().max(1e-3);
		  break;
		case RegularizationMethod::NORMALIZED_MIN_EIG:
		  singular_values = singular_values / singular_values.maxCoeff();
		  singular_values = singular_values.array().max(1e-3);
		  break;
		case RegularizationMethod::NORMALIZED_ELLIPSE:
		  // std::cout<<svd.singularValues()(1)<<std::endl;
		  if (singular_values(1) < 1e-3){
		  	singular_values = Eigen::Vector3d(1e-3, 1e-3, 1e-3);
		  }
		  else{          
			  singular_values = singular_values / singular_values(1);
			  // singular_values = singular_values.array().max(1e-3);
		  }
		  break;
    case RegularizationMethod::TEST:
      // singular_values = singular_values / singular_values(1) * 1e-2;
      break;
		case RegularizationMethod::NONE:
		  // do nothing
		  break;
		case RegularizationMethod::FROBENIUS:
		  std::cerr<< "should be implemented"<< std::endl;
		  abort();
	      }
	      // scales[i] = singular_values.cwiseSqrt();
	      // rotationsq[i] = input_rotationsq[i];
	      Eigen::Quaterniond q( (double)rotationsq[4*i+0], 
	      				(double)rotationsq[4*i+1], 
	      				(double)rotationsq[4*i+2], 
	      				(double)rotationsq[4*i+3]);
	      q = q.normalized();
	      covariances[i].setZero();
	      covariances[i].template block<3, 3>(0, 0) = q.toRotationMatrix() * singular_values.asDiagonal() * q.toRotationMatrix().transpose();
  }
  // std::cout << "Cloud size : " << target_->size() << "/cov size : " << covariances.size() << "/rots size : " << rotationsq.size()/4 << std::endl;

  // clock_t end_time = clock();
  // printf("Regularization time : %lf\n", (double)(end_time - start_time)/CLOCKS_PER_SEC);
}

}  // namespace fast_gicp

#endif
