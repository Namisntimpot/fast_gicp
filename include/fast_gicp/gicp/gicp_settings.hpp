#ifndef FAST_GICP_GICP_SETTINGS_HPP
#define FAST_GICP_GICP_SETTINGS_HPP

#include <array>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace fast_gicp {

constexpr int kLsqDof = 6;

enum class RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, PLANE, FROBENIUS, NORMALIZED_ELLIPSE, TEST };

enum class NeighborSearchMethod { DIRECT27, DIRECT7, DIRECT1, /* supported on only VGICP_CUDA */ DIRECT_RADIUS };

enum class VoxelAccumulationMode { ADDITIVE, ADDITIVE_WEIGHTED, MULTIPLICATIVE };

enum class SparseAnchorBalanceMode { NONE, BY_COUNT, BY_HESSIAN_TRACE };

struct ObservabilityConfig {
  bool enable_observability_check = false;
  double relative_eigenvalue_threshold = 1e-3;
  double absolute_eigenvalue_threshold = 1e-9;
  double ambiguity_score_threshold = 0.25;
  double auto_soft_prior_strength = 0.0;
  std::array<int, kLsqDof> hard_lock_mask{{0, 0, 0, 0, 0, 0}};
  std::array<int, kLsqDof> preferred_ambiguous_mask{{0, 0, 0, 0, 0, 0}};
  bool prefer_user_marked_dofs = true;
  bool enable_diagnostics = false;

  // Smooth observability prior (alternative to binary threshold)
  bool use_smooth_prior = false;
  double smooth_prior_falloff = 2.0;          // sigmoid steepness exponent
  double smooth_prior_max_strength = 0.1;     // max regularization relative to scale
  bool analyze_geometry_separately = false;    // eigendecompose geometry Hessian alone
  bool anchor_aware_regularization = false;    // reduce reg where anchors help
  int regularization_scale_mode = 0;           // 0=lambda_max, 1=geom_trace, 2=diag_mean
};

struct ObservabilityDiagnostics {
  Eigen::Matrix<double, kLsqDof, kLsqDof> raw_hessian = Eigen::Matrix<double, kLsqDof, kLsqDof>::Identity();
  Eigen::Matrix<double, kLsqDof, kLsqDof> regularized_hessian = Eigen::Matrix<double, kLsqDof, kLsqDof>::Identity();
  Eigen::Matrix<double, kLsqDof, 1> eigenvalues = Eigen::Matrix<double, kLsqDof, 1>::Zero();
  Eigen::Matrix<double, kLsqDof, 1> ambiguity_scores = Eigen::Matrix<double, kLsqDof, 1>::Zero();
  Eigen::Matrix<int, kLsqDof, 1> auto_suppressed_mask = Eigen::Matrix<int, kLsqDof, 1>::Zero();
  Eigen::Matrix<int, kLsqDof, 1> hard_lock_mask = Eigen::Matrix<int, kLsqDof, 1>::Zero();
  Eigen::Matrix<int, kLsqDof, 1> preferred_ambiguous_mask = Eigen::Matrix<int, kLsqDof, 1>::Zero();
  double condition_number = 1.0;
  int estimated_rank = kLsqDof;

  // Smooth prior diagnostics
  Eigen::Matrix<double, kLsqDof, 1> geometry_eigenvalues = Eigen::Matrix<double, kLsqDof, 1>::Zero();
  Eigen::Matrix<double, kLsqDof, 1> smooth_regularization_weights = Eigen::Matrix<double, kLsqDof, 1>::Zero();
  int geometry_estimated_rank = kLsqDof;
  double geometry_condition_number = 1.0;
};

struct ColorMatchingConfig {
  bool enable_color_matching = false;
  int geometric_candidate_count = 5;
  double color_weight = 0.0;
  double color_sigma = 32.0;
};

struct SparseAnchorConfig {
  double objective_weight = 1.0;
  SparseAnchorBalanceMode balance_mode = SparseAnchorBalanceMode::BY_HESSIAN_TRACE;
  double auto_balance_min = 1e-3;
  double auto_balance_max = 1e3;
};

struct AlignmentQualityConfig {
  bool enable_suggested_gating = false;
  bool require_converged = true;
  int min_correspondence_count = 0;
  int min_matched_count = 0;
  double min_matched_ratio = 0.0;
  double max_fitness_score = std::numeric_limits<double>::infinity();
  double max_final_cost = std::numeric_limits<double>::infinity();
  double max_normalized_cost_per_match = std::numeric_limits<double>::infinity();
  double max_mean_sq_distance = std::numeric_limits<double>::infinity();
  double max_median_sq_distance = std::numeric_limits<double>::infinity();
  double max_p90_sq_distance = std::numeric_limits<double>::infinity();
  double max_p95_sq_distance = std::numeric_limits<double>::infinity();
  int min_rank = 0;
  double max_condition_number = std::numeric_limits<double>::infinity();
  double max_ambiguity = std::numeric_limits<double>::infinity();
  std::array<double, kLsqDof> max_dof_ambiguity{
    {std::numeric_limits<double>::infinity(),
     std::numeric_limits<double>::infinity(),
     std::numeric_limits<double>::infinity(),
     std::numeric_limits<double>::infinity(),
     std::numeric_limits<double>::infinity(),
     std::numeric_limits<double>::infinity()}};
  double max_anchor_mean_residual = std::numeric_limits<double>::infinity();
  double max_anchor_p95_residual = std::numeric_limits<double>::infinity();
};

struct AlignmentQualityReport {
  bool valid = false;
  bool converged = false;
  bool gating_evaluated = false;
  bool suggested_accept = true;
  bool has_match_statistics = false;
  bool has_anchor_statistics = false;
  bool used_sparse_anchors = false;
  bool used_color_matching = false;
  std::string optimizer_type = "LevenbergMarquardt";
  int num_iterations = 0;
  int source_count = 0;
  int target_count = 0;
  int correspondence_count = 0;
  int matched_count = 0;
  int anchor_count = 0;
  int rank = kLsqDof;
  double fitness_score = std::numeric_limits<double>::quiet_NaN();
  double final_cost = std::numeric_limits<double>::quiet_NaN();
  double normalized_cost_per_match = std::numeric_limits<double>::quiet_NaN();
  double matched_ratio = std::numeric_limits<double>::quiet_NaN();
  double mean_sq_distance = std::numeric_limits<double>::quiet_NaN();
  double median_sq_distance = std::numeric_limits<double>::quiet_NaN();
  double p90_sq_distance = std::numeric_limits<double>::quiet_NaN();
  double p95_sq_distance = std::numeric_limits<double>::quiet_NaN();
  double condition_number = std::numeric_limits<double>::quiet_NaN();
  double max_ambiguity = std::numeric_limits<double>::quiet_NaN();
  double anchor_mean_residual = std::numeric_limits<double>::quiet_NaN();
  double anchor_p95_residual = std::numeric_limits<double>::quiet_NaN();
  double anchor_objective_weight = std::numeric_limits<double>::quiet_NaN();
  double anchor_auto_balance_factor = std::numeric_limits<double>::quiet_NaN();
  double anchor_effective_scale = std::numeric_limits<double>::quiet_NaN();
  double geometry_raw_cost = std::numeric_limits<double>::quiet_NaN();
  double anchor_raw_cost = std::numeric_limits<double>::quiet_NaN();
  double anchor_scaled_cost = std::numeric_limits<double>::quiet_NaN();
  double geometry_hessian_trace = std::numeric_limits<double>::quiet_NaN();
  double anchor_hessian_trace = std::numeric_limits<double>::quiet_NaN();
  bool anchor_balance_fallback_used = false;
  SparseAnchorBalanceMode anchor_balance_mode = SparseAnchorBalanceMode::NONE;
  Eigen::Matrix<double, kLsqDof, 1> ambiguity_scores = Eigen::Matrix<double, kLsqDof, 1>::Zero();
  Eigen::Matrix<int, kLsqDof, 1> auto_suppressed_mask = Eigen::Matrix<int, kLsqDof, 1>::Zero();
  Eigen::Matrix<int, kLsqDof, 1> hard_lock_mask = Eigen::Matrix<int, kLsqDof, 1>::Zero();
  Eigen::Matrix<double, kLsqDof, 1> smooth_regularization_weights = Eigen::Matrix<double, kLsqDof, 1>::Zero();
  int geometry_rank = kLsqDof;
  double geometry_condition_number = std::numeric_limits<double>::quiet_NaN();
  std::vector<std::string> rejection_reasons;
};
}

#endif
