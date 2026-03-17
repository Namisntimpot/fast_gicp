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
};

struct ColorMatchingConfig {
  bool enable_color_matching = false;
  int geometric_candidate_count = 5;
  double color_weight = 0.0;
  double color_sigma = 32.0;
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
  Eigen::Matrix<double, kLsqDof, 1> ambiguity_scores = Eigen::Matrix<double, kLsqDof, 1>::Zero();
  Eigen::Matrix<int, kLsqDof, 1> auto_suppressed_mask = Eigen::Matrix<int, kLsqDof, 1>::Zero();
  Eigen::Matrix<int, kLsqDof, 1> hard_lock_mask = Eigen::Matrix<int, kLsqDof, 1>::Zero();
  std::vector<std::string> rejection_reasons;
};
}

#endif
