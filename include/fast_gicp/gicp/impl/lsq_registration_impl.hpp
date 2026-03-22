#include <fast_gicp/gicp/lsq_registration.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <stdexcept>
#include <utility>

#include <boost/format.hpp>
#include <fast_gicp/so3/so3.hpp>

namespace fast_gicp {

namespace detail {

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

inline void append_unique_reason(std::vector<std::string>* reasons, const std::string& reason) {
  if (std::find(reasons->begin(), reasons->end(), reason) == reasons->end()) {
    reasons->push_back(reason);
  }
}

inline bool requires_match_statistics(const AlignmentQualityConfig& config) {
  return config.min_correspondence_count > 0 ||
         config.min_matched_count > 0 ||
         config.min_matched_ratio > 0.0 ||
         std::isfinite(config.max_normalized_cost_per_match) ||
         std::isfinite(config.max_mean_sq_distance) ||
         std::isfinite(config.max_median_sq_distance) ||
         std::isfinite(config.max_p90_sq_distance) ||
         std::isfinite(config.max_p95_sq_distance);
}

inline const char* optimizer_type_name(LSQ_OPTIMIZER_TYPE type) {
  switch (type) {
    case LSQ_OPTIMIZER_TYPE::GaussNewton:
      return "GaussNewton";
    case LSQ_OPTIMIZER_TYPE::LevenbergMarquardt:
      return "LevenbergMarquardt";
  }

  return "LevenbergMarquardt";
}

inline const std::array<const char*, kLsqDof>& ambiguity_reason_names() {
  static const std::array<const char*, kLsqDof> kReasonNames{{
    "rot_x_ambiguous",
    "rot_y_ambiguous",
    "rot_z_ambiguous",
    "trans_x_ambiguous",
    "trans_y_ambiguous",
    "trans_z_ambiguous",
  }};
  return kReasonNames;
}

inline const char* sparse_anchor_balance_mode_name(SparseAnchorBalanceMode mode) {
  switch (mode) {
    case SparseAnchorBalanceMode::NONE:
      return "NONE";
    case SparseAnchorBalanceMode::BY_COUNT:
      return "BY_COUNT";
    case SparseAnchorBalanceMode::BY_HESSIAN_TRACE:
      return "BY_HESSIAN_TRACE";
  }

  return "NONE";
}

inline double trace_on_unlocked_dofs(const Eigen::Matrix<double, kLsqDof, kLsqDof>& H, const std::array<int, kLsqDof>& hard_lock_mask) {
  double trace = 0.0;
  for (int i = 0; i < kLsqDof; i++) {
    if (!hard_lock_mask[i]) {
      trace += H(i, i);
    }
  }
  return trace;
}

}  // namespace detail

template <typename PointTarget, typename PointSource>
LsqRegistration<PointTarget, PointSource>::LsqRegistration() {
  this->reg_name_ = "LsqRegistration";
  max_iterations_ = 64;
  rotation_epsilon_ = 2e-3;
  transformation_epsilon_ = 5e-4;

  lsq_optimizer_type_ = LSQ_OPTIMIZER_TYPE::LevenbergMarquardt;
  lm_debug_print_ = false;
  lm_max_iterations_ = 10;
  lm_init_lambda_factor_ = 1e-9;
  lm_lambda_ = -1.0;

  final_hessian_.setIdentity();
  final_regularized_hessian_.setIdentity();
  use_sparse_anchors_ = false;
}

template <typename PointTarget, typename PointSource>
LsqRegistration<PointTarget, PointSource>::~LsqRegistration() {}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setRotationEpsilon(double eps) {
  rotation_epsilon_ = eps;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setInitialLambdaFactor(double init_lambda_factor) {
  lm_init_lambda_factor_ = init_lambda_factor;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setDebugPrint(bool lm_debug_print) {
  lm_debug_print_ = lm_debug_print;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setLSQOptimizerType(LSQ_OPTIMIZER_TYPE optimizer_type) {
  lsq_optimizer_type_ = optimizer_type;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setObservabilityConfig(const ObservabilityConfig& config) {
  observability_config_ = config;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setObservabilityCheck(bool enable) {
  observability_config_.enable_observability_check = enable;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setObservabilityEigenThresholds(double relative_threshold, double absolute_threshold) {
  observability_config_.relative_eigenvalue_threshold = relative_threshold;
  observability_config_.absolute_eigenvalue_threshold = absolute_threshold;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setAmbiguityScoreThreshold(double threshold) {
  observability_config_.ambiguity_score_threshold = threshold;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setAutoSoftPriorStrength(double strength) {
  observability_config_.auto_soft_prior_strength = strength;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setHardLockMask(const std::array<int, kLsqDof>& mask) {
  observability_config_.hard_lock_mask = mask;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setPreferredAmbiguousMask(const std::array<int, kLsqDof>& mask) {
  observability_config_.preferred_ambiguous_mask = mask;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setPreferUserMarkedDofs(bool prefer_user_marked_dofs) {
  observability_config_.prefer_user_marked_dofs = prefer_user_marked_dofs;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setEnableObservabilityDiagnostics(bool enable) {
  observability_config_.enable_diagnostics = enable;
}

template <typename PointTarget, typename PointSource>
const ObservabilityConfig& LsqRegistration<PointTarget, PointSource>::getObservabilityConfig() const {
  return observability_config_;
}

template <typename PointTarget, typename PointSource>
const ObservabilityDiagnostics& LsqRegistration<PointTarget, PointSource>::getObservabilityDiagnostics() const {
  return observability_diagnostics_;
}

template <typename PointTarget, typename PointSource>
const typename LsqRegistration<PointTarget, PointSource>::Matrix6& LsqRegistration<PointTarget, PointSource>::getFinalHessian() const {
  return final_hessian_;
}

template <typename PointTarget, typename PointSource>
const typename LsqRegistration<PointTarget, PointSource>::Matrix6& LsqRegistration<PointTarget, PointSource>::getFinalRegularizedHessian() const {
  return final_regularized_hessian_;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setAlignmentQualityConfig(const AlignmentQualityConfig& config) {
  alignment_quality_config_ = config;
}

template <typename PointTarget, typename PointSource>
const AlignmentQualityConfig& LsqRegistration<PointTarget, PointSource>::getAlignmentQualityConfig() const {
  return alignment_quality_config_;
}

template <typename PointTarget, typename PointSource>
const AlignmentQualityReport& LsqRegistration<PointTarget, PointSource>::getAlignmentQualityReport() const {
  return alignment_quality_report_;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setSparseAnchorConfig(const SparseAnchorConfig& config) {
  if (!std::isfinite(config.objective_weight) || config.objective_weight < 0.0) {
    throw std::invalid_argument("LsqRegistration: sparse anchor objective_weight must be finite and non-negative");
  }
  if (!std::isfinite(config.auto_balance_min) || !std::isfinite(config.auto_balance_max) ||
      config.auto_balance_min <= 0.0 || config.auto_balance_max <= 0.0) {
    throw std::invalid_argument("LsqRegistration: sparse anchor auto-balance limits must be finite and positive");
  }

  sparse_anchor_config_ = config;
  if (sparse_anchor_config_.auto_balance_min > sparse_anchor_config_.auto_balance_max) {
    std::swap(sparse_anchor_config_.auto_balance_min, sparse_anchor_config_.auto_balance_max);
  }
}

template <typename PointTarget, typename PointSource>
const SparseAnchorConfig& LsqRegistration<PointTarget, PointSource>::getSparseAnchorConfig() const {
  return sparse_anchor_config_;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setSparseAnchorObjectiveWeight(double weight) {
  SparseAnchorConfig config = sparse_anchor_config_;
  config.objective_weight = weight;
  setSparseAnchorConfig(config);
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setSparseAnchorBalanceMode(SparseAnchorBalanceMode mode) {
  sparse_anchor_config_.balance_mode = mode;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setSparseAnchorUsage(bool enable) {
  use_sparse_anchors_ = enable;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setSparseAnchorCorrespondences(
  const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& source_points,
  const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& target_points,
  const std::vector<double>& weights,
  const std::vector<double>& sigmas) {
  if (source_points.size() != target_points.size()) {
    throw std::invalid_argument("LsqRegistration: sparse anchor source/target size mismatch");
  }
  if (!weights.empty() && weights.size() != source_points.size()) {
    throw std::invalid_argument("LsqRegistration: sparse anchor weight size mismatch");
  }
  if (!sigmas.empty() && sigmas.size() != source_points.size()) {
    throw std::invalid_argument("LsqRegistration: sparse anchor sigma size mismatch");
  }

  sparse_anchor_source_points_ = source_points;
  sparse_anchor_target_points_ = target_points;
  sparse_anchor_weights_.assign(source_points.size(), 1.0);
  sparse_anchor_sigmas_.assign(source_points.size(), 1.0);

  if (!weights.empty()) {
    sparse_anchor_weights_ = weights;
  }
  if (!sigmas.empty()) {
    sparse_anchor_sigmas_ = sigmas;
  }

  use_sparse_anchors_ = !source_points.empty();
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::clearSparseAnchorCorrespondences() {
  sparse_anchor_source_points_.clear();
  sparse_anchor_target_points_.clear();
  sparse_anchor_weights_.clear();
  sparse_anchor_sigmas_.clear();
  use_sparse_anchors_ = false;
}

template <typename PointTarget, typename PointSource>
double LsqRegistration<PointTarget, PointSource>::evaluateCost(const Eigen::Matrix4f& relative_pose, Matrix6* H, Vector6* b) {
  PreparedLinearSystem system = build_linearized_system(Eigen::Isometry3f(relative_pose).cast<double>());
  if (H != nullptr) {
    *H = system.raw_hessian;
  }
  if (b != nullptr) {
    *b = system.gradient;
  }
  return system.cost;
}

template <typename PointTarget, typename PointSource>
int LsqRegistration<PointTarget, PointSource>::current_geometric_term_count() const {
  return input_ ? static_cast<int>(input_->size()) : 0;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  Eigen::Isometry3d x0 = Eigen::Isometry3d(guess.template cast<double>());

  lm_lambda_ = -1.0;
  converged_ = false;
  nr_iterations_ = 0;
  alignment_quality_report_ = AlignmentQualityReport();

  if (lm_debug_print_) {
    std::cout << "********************************************" << std::endl;
    std::cout << "***************** optimize *****************" << std::endl;
    std::cout << "********************************************" << std::endl;
  }

  int iterations_performed = 0;
  for (int i = 0; i < max_iterations_ && !converged_; i++) {
    iterations_performed = i + 1;
    Eigen::Isometry3d delta;
    if (!step_optimize(x0, delta)) {
      std::cerr << "lm not converged!!" << std::endl;
      break;
    }

    converged_ = is_converged(delta);
  }

  nr_iterations_ = iterations_performed;
  final_transformation_ = x0.cast<float>().matrix();

  PreparedLinearSystem final_system = build_linearized_system(x0, true);
  final_hessian_ = final_system.raw_hessian;
  final_regularized_hessian_ = final_system.regularized_hessian;
  fill_alignment_quality_report(final_system, x0);

  pcl::transformPointCloud(*input_, output, final_transformation_);
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::is_converged(const Eigen::Isometry3d& delta) const {
  Eigen::Matrix3d R = delta.linear() - Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = delta.translation();

  Eigen::Matrix3d r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
  Eigen::Vector3d t_delta = 1.0 / transformation_epsilon_ * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_optimize(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  switch (lsq_optimizer_type_) {
    case LSQ_OPTIMIZER_TYPE::LevenbergMarquardt:
      return step_lm(x0, delta);
    case LSQ_OPTIMIZER_TYPE::GaussNewton:
      return step_gn(x0, delta);
  }

  return step_lm(x0, delta);
}

template <typename PointTarget, typename PointSource>
typename LsqRegistration<PointTarget, PointSource>::PreparedLinearSystem LsqRegistration<PointTarget, PointSource>::build_linearized_system(
  const Eigen::Isometry3d& trans,
  bool force_observability_analysis) {
  PreparedLinearSystem system;
  Matrix6 geometry_hessian = Matrix6::Zero();
  Vector6 geometry_gradient = Vector6::Zero();
  Matrix6 anchor_hessian = Matrix6::Zero();
  Vector6 anchor_gradient = Vector6::Zero();

  system.geometry_raw_cost = linearize(trans, &geometry_hessian, &geometry_gradient);
  system.anchor_raw_cost = sparse_anchor_cost(trans, &anchor_hessian, &anchor_gradient);

  const Matrix6 geometry_hessian_sym = 0.5 * (geometry_hessian + geometry_hessian.transpose());
  const Matrix6 anchor_hessian_sym = 0.5 * (anchor_hessian + anchor_hessian.transpose());

  system.geometry_hessian_trace = detail::trace_on_unlocked_dofs(geometry_hessian_sym, observability_config_.hard_lock_mask);
  system.anchor_hessian_trace = detail::trace_on_unlocked_dofs(anchor_hessian_sym, observability_config_.hard_lock_mask);
  system.anchor_balance_factor = 1.0;
  system.anchor_effective_scale = 0.0;
  system.anchor_balance_fallback_used = false;

  if (use_sparse_anchors_ && !sparse_anchor_source_points_.empty()) {
    const auto clamp_factor = [this](double factor) {
      return std::min(
        sparse_anchor_config_.auto_balance_max,
        std::max(sparse_anchor_config_.auto_balance_min, factor));
    };

    switch (sparse_anchor_config_.balance_mode) {
      case SparseAnchorBalanceMode::NONE:
        system.anchor_balance_factor = 1.0;
        break;
      case SparseAnchorBalanceMode::BY_COUNT: {
        const int anchor_count = static_cast<int>(sparse_anchor_source_points_.size());
        const int geometric_term_count = current_geometric_term_count();
        if (anchor_count > 0 && geometric_term_count > 0) {
          system.anchor_balance_factor =
            clamp_factor(static_cast<double>(geometric_term_count) / static_cast<double>(anchor_count));
        } else {
          system.anchor_balance_factor = 1.0;
          system.anchor_balance_fallback_used = true;
        }
        break;
      }
      case SparseAnchorBalanceMode::BY_HESSIAN_TRACE: {
        if (std::isfinite(system.geometry_hessian_trace) && std::isfinite(system.anchor_hessian_trace) &&
            system.geometry_hessian_trace > 1e-12 && system.anchor_hessian_trace > 1e-12) {
          system.anchor_balance_factor = clamp_factor(system.geometry_hessian_trace / system.anchor_hessian_trace);
        } else {
          system.anchor_balance_factor = 1.0;
          system.anchor_balance_fallback_used = true;
        }
        break;
      }
    }

    system.anchor_effective_scale = sparse_anchor_config_.objective_weight * system.anchor_balance_factor;
  }

  system.cost = system.geometry_raw_cost + system.anchor_effective_scale * system.anchor_raw_cost;

  Matrix6 H = geometry_hessian + system.anchor_effective_scale * anchor_hessian;
  Vector6 b = geometry_gradient + system.anchor_effective_scale * anchor_gradient;

  system.raw_hessian = 0.5 * (H + H.transpose());
  system.regularized_hessian = system.raw_hessian;
  system.gradient = b;

  const bool analyze =
    force_observability_analysis || observability_config_.enable_observability_check || observability_config_.enable_diagnostics;
  if (!analyze) {
    return system;
  }

  // --- Determine which Hessian to analyze for degeneracy ---
  const Matrix6 analysis_hessian = observability_config_.analyze_geometry_separately
    ? geometry_hessian_sym
    : system.raw_hessian;

  Eigen::SelfAdjointEigenSolver<Matrix6> eigen_solver(analysis_hessian);
  if (eigen_solver.info() != Eigen::Success) {
    store_observability_diagnostics(system);
    return system;
  }

  const Vector6 analysis_eigenvalues = eigen_solver.eigenvalues();
  const Matrix6 analysis_eigenvectors = eigen_solver.eigenvectors();

  // Fill geometry-specific diagnostics when analyzing geometry separately
  if (observability_config_.analyze_geometry_separately) {
    system.geometry_eigenvalues = analysis_eigenvalues;
    // Also eigendecompose the combined Hessian for reporting
    Eigen::SelfAdjointEigenSolver<Matrix6> combined_solver(system.raw_hessian);
    if (combined_solver.info() == Eigen::Success) {
      system.eigenvalues = combined_solver.eigenvalues();
    }
  } else {
    system.eigenvalues = analysis_eigenvalues;
    system.geometry_eigenvalues = analysis_eigenvalues;  // same when not separate
  }

  const double lambda_max = std::max(system.eigenvalues.maxCoeff(), observability_config_.absolute_eigenvalue_threshold);
  const double analysis_lambda_max = std::max(analysis_eigenvalues.maxCoeff(), observability_config_.absolute_eigenvalue_threshold);
  const double eigen_threshold = std::max(
    observability_config_.absolute_eigenvalue_threshold,
    observability_config_.relative_eigenvalue_threshold * analysis_lambda_max);

  // Compute rank and condition number from analysis Hessian
  std::vector<int> degenerate_columns;
  degenerate_columns.reserve(kLsqDof);
  int analysis_rank = 0;
  double smallest_kept = std::numeric_limits<double>::infinity();
  for (int i = 0; i < kLsqDof; i++) {
    if (analysis_eigenvalues[i] > eigen_threshold) {
      analysis_rank++;
      smallest_kept = std::min(smallest_kept, analysis_eigenvalues[i]);
    } else {
      degenerate_columns.push_back(i);
    }
  }

  if (observability_config_.analyze_geometry_separately) {
    system.geometry_estimated_rank = analysis_rank;
    system.geometry_condition_number = (analysis_rank == 0 || !std::isfinite(smallest_kept))
      ? std::numeric_limits<double>::infinity()
      : analysis_lambda_max / smallest_kept;
    // Combined rank/condition from combined eigenvalues
    system.estimated_rank = 0;
    double combined_smallest = std::numeric_limits<double>::infinity();
    for (int i = 0; i < kLsqDof; i++) {
      if (system.eigenvalues[i] > eigen_threshold) {
        system.estimated_rank++;
        combined_smallest = std::min(combined_smallest, system.eigenvalues[i]);
      }
    }
    system.condition_number = (system.estimated_rank == 0 || !std::isfinite(combined_smallest))
      ? std::numeric_limits<double>::infinity()
      : lambda_max / combined_smallest;
  } else {
    system.estimated_rank = analysis_rank;
    system.geometry_estimated_rank = analysis_rank;
    if (analysis_rank == 0 || !std::isfinite(smallest_kept)) {
      system.condition_number = std::numeric_limits<double>::infinity();
      system.geometry_condition_number = std::numeric_limits<double>::infinity();
    } else {
      system.condition_number = analysis_lambda_max / smallest_kept;
      system.geometry_condition_number = system.condition_number;
    }
  }

  // Compute ambiguity scores from degenerate eigenvectors of analysis Hessian
  system.ambiguity_scores.setZero();
  for (int column : degenerate_columns) {
    system.ambiguity_scores.array() += analysis_eigenvectors.col(column).array().square();
  }

  // === SMOOTH PRIOR PATH ===
  if (observability_config_.enable_observability_check &&
      observability_config_.use_smooth_prior) {

    // Smooth per-eigenvalue weight: w(i) = 1 / (1 + (ratio/threshold)^falloff)
    // Small eigenvalue -> ratio close to 0 -> weight close to 1 (more regularization)
    // Large eigenvalue -> ratio >> threshold -> weight close to 0 (no regularization)
    Vector6 per_eigval_weight;
    for (int i = 0; i < kLsqDof; i++) {
      double ratio = analysis_eigenvalues[i] / analysis_lambda_max;
      double scaled = std::pow(
        std::max(ratio, 1e-15) / observability_config_.relative_eigenvalue_threshold,
        observability_config_.smooth_prior_falloff);
      per_eigval_weight[i] = 1.0 / (1.0 + scaled);
    }

    // Transform eigenvalue-space weights to DOF-space weights
    // dof_weight(j) = sum_i [ per_eigval_weight(i) * eigvec(j,i)^2 ]
    Vector6 dof_weights = Vector6::Zero();
    for (int i = 0; i < kLsqDof; i++) {
      dof_weights.array() += per_eigval_weight[i] * analysis_eigenvectors.col(i).array().square();
    }

    // Determine regularization scale
    double reg_scale;
    switch (observability_config_.regularization_scale_mode) {
      case 1:
        // Scale relative to geometry Hessian trace (more stable)
        reg_scale = std::max(system.geometry_hessian_trace / static_cast<double>(kLsqDof), 1e-12);
        break;
      case 2:
        // Scale relative to diagonal mean of combined Hessian
        reg_scale = std::max(system.raw_hessian.diagonal().mean(), 1e-12);
        break;
      default:
        // Scale relative to lambda_max (original-like behavior)
        reg_scale = std::max(analysis_lambda_max, 1e-12);
        break;
    }

    // Anchor-aware reduction: if anchors provide information in degenerate directions,
    // reduce the soft prior there
    if (observability_config_.anchor_aware_regularization &&
        use_sparse_anchors_ && !sparse_anchor_source_points_.empty() &&
        system.anchor_effective_scale > 0.0) {
      Eigen::SelfAdjointEigenSolver<Matrix6> anchor_solver(anchor_hessian_sym);
      if (anchor_solver.info() == Eigen::Success) {
        const Vector6 anchor_ev = anchor_solver.eigenvalues();
        const Matrix6 anchor_eigvecs = anchor_solver.eigenvectors();

        // Compute per-DOF anchor contribution normalized to analysis scale
        Vector6 anchor_dof_strength = Vector6::Zero();
        for (int i = 0; i < kLsqDof; i++) {
          if (anchor_ev[i] > 1e-12) {
            double anchor_ratio = anchor_ev[i] * system.anchor_effective_scale / analysis_lambda_max;
            anchor_dof_strength.array() +=
              std::min(anchor_ratio, 1.0) * anchor_eigvecs.col(i).array().square();
          }
        }

        // Reduce regularization where anchors provide information
        for (int i = 0; i < kLsqDof; i++) {
          double reduction = 1.0 - std::min(anchor_dof_strength[i], 1.0);
          dof_weights[i] *= reduction;
        }
      }
    }

    // Apply smooth regularization
    const double max_reg = observability_config_.smooth_prior_max_strength * reg_scale;
    for (int i = 0; i < kLsqDof; i++) {
      if (observability_config_.hard_lock_mask[i]) continue;
      double reg_amount = max_reg * dof_weights[i];
      system.regularized_hessian(i, i) += reg_amount;
      system.smooth_regularization_weights[i] = reg_amount;
      if (dof_weights[i] > 0.01) {
        system.auto_suppressed_mask[i] = 1;
      }
    }

  } else if (observability_config_.enable_observability_check &&
             observability_config_.auto_soft_prior_strength > 0.0 &&
             !degenerate_columns.empty()) {
    // === ORIGINAL BINARY OBSERVABILITY CHECK (backward compatible) ===
    std::vector<int> preferred_indices;
    std::vector<int> other_indices;
    preferred_indices.reserve(kLsqDof);
    other_indices.reserve(kLsqDof);

    for (int i = 0; i < kLsqDof; i++) {
      if (observability_config_.prefer_user_marked_dofs && observability_config_.preferred_ambiguous_mask[i]) {
        preferred_indices.push_back(i);
      } else {
        other_indices.push_back(i);
      }
    }

    auto sort_by_score = [&system](std::vector<int>* indices) {
      std::sort(indices->begin(), indices->end(), [&system](int lhs, int rhs) {
        return system.ambiguity_scores[lhs] > system.ambiguity_scores[rhs];
      });
    };
    sort_by_score(&preferred_indices);
    sort_by_score(&other_indices);

    std::vector<int> selected;
    selected.reserve(degenerate_columns.size());
    auto append_candidates = [&](const std::vector<int>& indices, bool require_threshold) {
      for (int idx : indices) {
        if (selected.size() >= degenerate_columns.size()) {
          break;
        }
        if (system.ambiguity_scores[idx] <= 0.0) {
          continue;
        }
        if (require_threshold && system.ambiguity_scores[idx] < observability_config_.ambiguity_score_threshold) {
          continue;
        }
        if (std::find(selected.begin(), selected.end(), idx) == selected.end()) {
          selected.push_back(idx);
        }
      }
    };

    append_candidates(preferred_indices, true);
    append_candidates(other_indices, true);
    append_candidates(preferred_indices, false);
    append_candidates(other_indices, false);

    for (int idx : selected) {
      system.auto_suppressed_mask[idx] = 1;
      system.regularized_hessian(idx, idx) += observability_config_.auto_soft_prior_strength * analysis_lambda_max * system.ambiguity_scores[idx];
    }
  }

  store_observability_diagnostics(system);
  return system;
}

template <typename PointTarget, typename PointSource>
typename LsqRegistration<PointTarget, PointSource>::Vector6 LsqRegistration<PointTarget, PointSource>::solve_linearized_system(
  const Matrix6& H,
  const Vector6& b,
  const std::array<int, kLsqDof>& hard_lock_mask) const {
  Vector6 solution = Vector6::Zero();

  std::vector<int> active_indices;
  active_indices.reserve(kLsqDof);
  for (int i = 0; i < kLsqDof; i++) {
    if (!hard_lock_mask[i]) {
      active_indices.push_back(i);
    }
  }

  if (active_indices.empty()) {
    return solution;
  }

  Eigen::MatrixXd H_active(active_indices.size(), active_indices.size());
  Eigen::VectorXd b_active(active_indices.size());
  for (int r = 0; r < active_indices.size(); r++) {
    b_active[r] = b[active_indices[r]];
    for (int c = 0; c < active_indices.size(); c++) {
      H_active(r, c) = H(active_indices[r], active_indices[c]);
    }
  }

  Eigen::VectorXd d_active;
  Eigen::LDLT<Eigen::MatrixXd> solver(H_active);
  if (solver.info() == Eigen::Success) {
    d_active = solver.solve(-b_active);
    if (solver.info() != Eigen::Success) {
      d_active = H_active.completeOrthogonalDecomposition().solve(-b_active);
    }
  } else {
    d_active = H_active.completeOrthogonalDecomposition().solve(-b_active);
  }

  for (int i = 0; i < active_indices.size(); i++) {
    solution[active_indices[i]] = d_active[i];
  }

  return solution;
}

template <typename PointTarget, typename PointSource>
double LsqRegistration<PointTarget, PointSource>::sparse_anchor_cost(const Eigen::Isometry3d& trans, Matrix6* H, Vector6* b) const {
  if (!use_sparse_anchors_ || sparse_anchor_source_points_.empty()) {
    return 0.0;
  }

  double cost = 0.0;
  for (int i = 0; i < sparse_anchor_source_points_.size(); i++) {
    const Eigen::Vector4d source_point(sparse_anchor_source_points_[i].x(), sparse_anchor_source_points_[i].y(), sparse_anchor_source_points_[i].z(), 1.0);
    const Eigen::Vector4d target_point(sparse_anchor_target_points_[i].x(), sparse_anchor_target_points_[i].y(), sparse_anchor_target_points_[i].z(), 1.0);
    const Eigen::Vector4d transformed_source = trans * source_point;
    const Eigen::Vector4d error = target_point - transformed_source;
    const double sigma = sparse_anchor_sigmas_.empty() ? 1.0 : std::max(sparse_anchor_sigmas_[i], 1e-9);
    const double weight = (sparse_anchor_weights_.empty() ? 1.0 : sparse_anchor_weights_[i]) / (sigma * sigma);

    cost += weight * error.head<3>().squaredNorm();

    if (H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, kLsqDof> dtdx0 = Eigen::Matrix<double, 4, kLsqDof>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transformed_source.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    *H += weight * dtdx0.transpose() * dtdx0;
    *b += weight * dtdx0.transpose() * error;
  }

  return cost;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::store_observability_diagnostics(const PreparedLinearSystem& system) {
  observability_diagnostics_.raw_hessian = system.raw_hessian;
  observability_diagnostics_.regularized_hessian = system.regularized_hessian;
  observability_diagnostics_.eigenvalues = system.eigenvalues;
  observability_diagnostics_.ambiguity_scores = system.ambiguity_scores;
  observability_diagnostics_.condition_number = system.condition_number;
  observability_diagnostics_.estimated_rank = system.estimated_rank;
  observability_diagnostics_.geometry_eigenvalues = system.geometry_eigenvalues;
  observability_diagnostics_.smooth_regularization_weights = system.smooth_regularization_weights;
  observability_diagnostics_.geometry_estimated_rank = system.geometry_estimated_rank;
  observability_diagnostics_.geometry_condition_number = system.geometry_condition_number;

  for (int i = 0; i < kLsqDof; i++) {
    observability_diagnostics_.auto_suppressed_mask[i] = system.auto_suppressed_mask[i];
    observability_diagnostics_.hard_lock_mask[i] = observability_config_.hard_lock_mask[i];
    observability_diagnostics_.preferred_ambiguous_mask[i] = observability_config_.preferred_ambiguous_mask[i];
  }
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::fill_alignment_quality_report(
  const PreparedLinearSystem& system,
  const Eigen::Isometry3d& final_pose) {
  AlignmentQualityReport report;
  report.valid = true;
  report.converged = converged_;
  report.gating_evaluated = false;
  report.suggested_accept = true;
  report.used_sparse_anchors = use_sparse_anchors_ && !sparse_anchor_source_points_.empty();
  report.used_color_matching = false;
  report.optimizer_type = detail::optimizer_type_name(lsq_optimizer_type_);
  report.num_iterations = nr_iterations_;
  report.source_count = input_ ? static_cast<int>(input_->size()) : 0;
  report.target_count = target_ ? static_cast<int>(target_->size()) : 0;
  report.rank = system.estimated_rank;
  report.final_cost = system.cost;
  report.condition_number = system.condition_number;
  report.ambiguity_scores = system.ambiguity_scores;
  report.max_ambiguity = system.ambiguity_scores.maxCoeff();
  report.anchor_objective_weight = sparse_anchor_config_.objective_weight;
  report.anchor_balance_mode = sparse_anchor_config_.balance_mode;
  report.anchor_auto_balance_factor = system.anchor_balance_factor;
  report.anchor_effective_scale = system.anchor_effective_scale;
  report.geometry_raw_cost = system.geometry_raw_cost;
  report.anchor_raw_cost = system.anchor_raw_cost;
  report.anchor_scaled_cost = system.anchor_effective_scale * system.anchor_raw_cost;
  report.geometry_hessian_trace = system.geometry_hessian_trace;
  report.anchor_hessian_trace = system.anchor_hessian_trace;
  report.anchor_balance_fallback_used = system.anchor_balance_fallback_used;
  report.auto_suppressed_mask = Eigen::Map<const Eigen::Matrix<int, kLsqDof, 1>>(system.auto_suppressed_mask.data());
  report.hard_lock_mask = Eigen::Map<const Eigen::Matrix<int, kLsqDof, 1>>(observability_config_.hard_lock_mask.data());
  report.smooth_regularization_weights = system.smooth_regularization_weights;
  report.geometry_rank = system.geometry_estimated_rank;
  report.geometry_condition_number = system.geometry_condition_number;

  if (input_ != nullptr && target_ != nullptr && !input_->empty() && !target_->empty()) {
    report.fitness_score = this->getFitnessScore();
  }

  collect_alignment_quality_metrics(&report, final_pose);
  if (report.source_count > 0 && !std::isfinite(report.matched_ratio)) {
    report.matched_ratio = static_cast<double>(report.matched_count) / static_cast<double>(report.source_count);
  }

  if (report.used_sparse_anchors) {
    collect_anchor_quality_metrics(&report, final_pose);
  }

  if (report.has_match_statistics && report.correspondence_count > 0) {
    report.normalized_cost_per_match = report.final_cost / static_cast<double>(report.correspondence_count);
  }

  evaluate_quality_gating(&report);
  alignment_quality_report_ = report;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::collect_anchor_quality_metrics(
  AlignmentQualityReport* report,
  const Eigen::Isometry3d& final_pose) const {
  if (!report->used_sparse_anchors || sparse_anchor_source_points_.empty()) {
    return;
  }

  std::vector<double> residuals;
  residuals.reserve(sparse_anchor_source_points_.size());
  for (std::size_t i = 0; i < sparse_anchor_source_points_.size(); i++) {
    const Eigen::Vector3d transformed = final_pose * sparse_anchor_source_points_[i];
    residuals.push_back((sparse_anchor_target_points_[i] - transformed).norm());
  }

  if (residuals.empty()) {
    return;
  }

  std::sort(residuals.begin(), residuals.end());
  report->has_anchor_statistics = true;
  report->anchor_count = static_cast<int>(residuals.size());
  report->anchor_mean_residual =
    std::accumulate(residuals.begin(), residuals.end(), 0.0) / static_cast<double>(residuals.size());
  report->anchor_p95_residual = detail::percentile_from_sorted(residuals, 0.95);
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::evaluate_quality_gating(AlignmentQualityReport* report) const {
  report->gating_evaluated = alignment_quality_config_.enable_suggested_gating;
  report->suggested_accept = true;
  report->rejection_reasons.clear();

  if (!alignment_quality_config_.enable_suggested_gating) {
    return;
  }

  const auto reject = [report](const std::string& reason) {
    detail::append_unique_reason(&report->rejection_reasons, reason);
    report->suggested_accept = false;
  };

  if (alignment_quality_config_.require_converged && !report->converged) {
    reject("not_converged");
  }

  const bool needs_match_stats = detail::requires_match_statistics(alignment_quality_config_);
  if (needs_match_stats && !report->has_match_statistics) {
    reject("required_match_statistics_unavailable");
  }

  if (report->has_match_statistics) {
    if (report->correspondence_count < alignment_quality_config_.min_correspondence_count) {
      reject("too_few_correspondences");
    }
    if (report->matched_count < alignment_quality_config_.min_matched_count) {
      reject("too_few_matches");
    }
    if (report->matched_ratio < alignment_quality_config_.min_matched_ratio) {
      reject("low_match_ratio");
    }
    if (std::isfinite(alignment_quality_config_.max_normalized_cost_per_match) &&
        (!std::isfinite(report->normalized_cost_per_match) ||
         report->normalized_cost_per_match > alignment_quality_config_.max_normalized_cost_per_match)) {
      reject("normalized_cost_too_high");
    }
    if (std::isfinite(alignment_quality_config_.max_mean_sq_distance) &&
        (!std::isfinite(report->mean_sq_distance) || report->mean_sq_distance > alignment_quality_config_.max_mean_sq_distance)) {
      reject("mean_sq_distance_too_high");
    }
    if (std::isfinite(alignment_quality_config_.max_median_sq_distance) &&
        (!std::isfinite(report->median_sq_distance) || report->median_sq_distance > alignment_quality_config_.max_median_sq_distance)) {
      reject("median_sq_distance_too_high");
    }
    if (std::isfinite(alignment_quality_config_.max_p90_sq_distance) &&
        (!std::isfinite(report->p90_sq_distance) || report->p90_sq_distance > alignment_quality_config_.max_p90_sq_distance)) {
      reject("p90_sq_distance_too_high");
    }
    if (std::isfinite(alignment_quality_config_.max_p95_sq_distance) &&
        (!std::isfinite(report->p95_sq_distance) || report->p95_sq_distance > alignment_quality_config_.max_p95_sq_distance)) {
      reject("p95_sq_distance_too_high");
    }
  }

  if (std::isfinite(alignment_quality_config_.max_fitness_score) &&
      (!std::isfinite(report->fitness_score) || report->fitness_score > alignment_quality_config_.max_fitness_score)) {
    reject("fitness_too_high");
  }
  if (std::isfinite(alignment_quality_config_.max_final_cost) &&
      (!std::isfinite(report->final_cost) || report->final_cost > alignment_quality_config_.max_final_cost)) {
    reject("final_cost_too_high");
  }
  if (report->rank < alignment_quality_config_.min_rank) {
    reject("rank_too_low");
  }
  if (std::isfinite(alignment_quality_config_.max_condition_number) &&
      (!std::isfinite(report->condition_number) || report->condition_number > alignment_quality_config_.max_condition_number)) {
    reject("condition_number_too_high");
  }
  if (std::isfinite(alignment_quality_config_.max_ambiguity) &&
      (!std::isfinite(report->max_ambiguity) || report->max_ambiguity > alignment_quality_config_.max_ambiguity)) {
    reject("max_ambiguity_too_high");
  }

  const auto& dof_reason_names = detail::ambiguity_reason_names();
  for (int i = 0; i < kLsqDof; i++) {
    if (std::isfinite(alignment_quality_config_.max_dof_ambiguity[i]) &&
        (!std::isfinite(report->ambiguity_scores[i]) || report->ambiguity_scores[i] > alignment_quality_config_.max_dof_ambiguity[i])) {
      reject(dof_reason_names[i]);
    }
  }

  if (report->used_sparse_anchors) {
    if (!report->has_anchor_statistics &&
        (std::isfinite(alignment_quality_config_.max_anchor_mean_residual) ||
         std::isfinite(alignment_quality_config_.max_anchor_p95_residual))) {
      reject("required_anchor_statistics_unavailable");
    }
    if (report->has_anchor_statistics) {
      if (std::isfinite(alignment_quality_config_.max_anchor_mean_residual) &&
          (!std::isfinite(report->anchor_mean_residual) ||
           report->anchor_mean_residual > alignment_quality_config_.max_anchor_mean_residual)) {
        reject("anchor_mean_residual_too_high");
      }
      if (std::isfinite(alignment_quality_config_.max_anchor_p95_residual) &&
          (!std::isfinite(report->anchor_p95_residual) ||
           report->anchor_p95_residual > alignment_quality_config_.max_anchor_p95_residual)) {
        reject("anchor_p95_residual_too_high");
      }
    }
  }
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::collect_alignment_quality_metrics(
  AlignmentQualityReport* report,
  const Eigen::Isometry3d& final_pose) const {
  (void)report;
  (void)final_pose;
}

template <typename PointTarget, typename PointSource>
Eigen::Isometry3d LsqRegistration<PointTarget, PointSource>::apply_hard_locks_to_pose(
  const Eigen::Isometry3d& previous_pose,
  const Eigen::Isometry3d& candidate_pose) const {
  Eigen::Isometry3d constrained_pose = candidate_pose;
  bool has_rotation_lock = false;
  for (int axis = 0; axis < 3; axis++) {
    has_rotation_lock = has_rotation_lock || observability_config_.hard_lock_mask[axis];
  }

  if (has_rotation_lock) {
    Eigen::Vector3d previous_euler = previous_pose.linear().eulerAngles(0, 1, 2);
    Eigen::Vector3d candidate_euler = candidate_pose.linear().eulerAngles(0, 1, 2);
    for (int axis = 0; axis < 3; axis++) {
      if (observability_config_.hard_lock_mask[axis]) {
        candidate_euler[axis] = previous_euler[axis];
      }
    }

    constrained_pose.linear() =
      (Eigen::AngleAxisd(candidate_euler[0], Eigen::Vector3d::UnitX()) *
       Eigen::AngleAxisd(candidate_euler[1], Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(candidate_euler[2], Eigen::Vector3d::UnitZ()))
        .toRotationMatrix();
  }

  for (int axis = 0; axis < 3; axis++) {
    if (observability_config_.hard_lock_mask[axis + 3]) {
      constrained_pose.translation()[axis] = previous_pose.translation()[axis];
    }
  }

  return constrained_pose;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_gn(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  PreparedLinearSystem system = build_linearized_system(x0);
  Vector6 d = solve_linearized_system(system.regularized_hessian, system.gradient, observability_config_.hard_lock_mask);

  delta.setIdentity();
  delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
  delta.translation() = d.tail<3>();

  x0 = apply_hard_locks_to_pose(x0, delta * x0);
  final_hessian_ = system.raw_hessian;
  final_regularized_hessian_ = system.regularized_hessian;

  return true;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_lm(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  PreparedLinearSystem system = build_linearized_system(x0);
  double y0 = system.cost;

  if (lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * system.regularized_hessian.diagonal().array().abs().maxCoeff();
  }

  double nu = 2.0;
  for (int i = 0; i < lm_max_iterations_; i++) {
    Matrix6 H_lm = system.regularized_hessian;
    for (int dof = 0; dof < kLsqDof; dof++) {
      if (!observability_config_.hard_lock_mask[dof]) {
        H_lm(dof, dof) += lm_lambda_;
      }
    }

    Vector6 d = solve_linearized_system(H_lm, system.gradient, observability_config_.hard_lock_mask);

    delta.setIdentity();
    delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
    delta.translation() = d.tail<3>();

    Eigen::Isometry3d xi = apply_hard_locks_to_pose(x0, delta * x0);
    double yi = compute_error(xi) + system.anchor_effective_scale * sparse_anchor_cost(xi);

    Vector6 damping = Vector6::Zero();
    for (int dof = 0; dof < kLsqDof; dof++) {
      if (!observability_config_.hard_lock_mask[dof]) {
        damping[dof] = lm_lambda_ * d[dof];
      }
    }
    const double denom = d.dot(damping - system.gradient);
    const double rho = std::abs(denom) < 1e-12 ? -1.0 : (y0 - yi) / denom;

    if (lm_debug_print_) {
      if (i == 0) {
        std::cout << boost::format("--- LM optimization ---\n%5s %15s %15s %15s %15s %15s %5s\n") % "i" % "y0" % "yi" % "rho" % "lambda" % "|delta|" % "dec";
      }
      char dec = rho > 0.0 ? 'x' : ' ';
      std::cout << boost::format("%5d %15g %15g %15g %15g %15g %5c") % i % y0 % yi % rho % lm_lambda_ % d.norm() % dec << std::endl;
    }

    if (rho < 0) {
      if (is_converged(delta)) {
        return true;
      }

      lm_lambda_ = nu * lm_lambda_;
      nu = 2 * nu;
      continue;
    }

    x0 = xi;
    lm_lambda_ = lm_lambda_ * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
    final_hessian_ = system.raw_hessian;
    final_regularized_hessian_ = system.regularized_hessian;
    return true;
  }

  return false;
}

}  // namespace fast_gicp
