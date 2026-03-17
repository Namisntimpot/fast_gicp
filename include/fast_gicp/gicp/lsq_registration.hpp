#ifndef FAST_GICP_LSQ_REGISTRATION_HPP
#define FAST_GICP_LSQ_REGISTRATION_HPP

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/QR>

#include <array>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

enum class LSQ_OPTIMIZER_TYPE { GaussNewton, LevenbergMarquardt };

template<typename PointSource, typename PointTarget>
class LsqRegistration : public pcl::Registration<PointSource, PointTarget, float> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;
  using Matrix6 = Eigen::Matrix<double, kLsqDof, kLsqDof>;
  using Vector6 = Eigen::Matrix<double, kLsqDof, 1>;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  using Ptr = pcl::shared_ptr<LsqRegistration<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const LsqRegistration<PointSource, PointTarget>>;
#else
  using Ptr = boost::shared_ptr<LsqRegistration<PointSource, PointTarget>>;
  using ConstPtr = boost::shared_ptr<const LsqRegistration<PointSource, PointTarget>>;
#endif

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::converged_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LsqRegistration();
  virtual ~LsqRegistration();

  void setRotationEpsilon(double eps);
  void setInitialLambdaFactor(double init_lambda_factor);
  void setDebugPrint(bool lm_debug_print);
  void setLSQOptimizerType(LSQ_OPTIMIZER_TYPE optimizer_type);
  void setObservabilityConfig(const ObservabilityConfig& config);
  void setObservabilityCheck(bool enable);
  void setObservabilityEigenThresholds(double relative_threshold, double absolute_threshold);
  void setAmbiguityScoreThreshold(double threshold);
  void setAutoSoftPriorStrength(double strength);
  void setHardLockMask(const std::array<int, kLsqDof>& mask);
  void setPreferredAmbiguousMask(const std::array<int, kLsqDof>& mask);
  void setPreferUserMarkedDofs(bool prefer_user_marked_dofs);
  void setEnableObservabilityDiagnostics(bool enable);
  const ObservabilityConfig& getObservabilityConfig() const;
  const ObservabilityDiagnostics& getObservabilityDiagnostics() const;
  const Matrix6& getFinalRegularizedHessian() const;
  void setAlignmentQualityConfig(const AlignmentQualityConfig& config);
  const AlignmentQualityConfig& getAlignmentQualityConfig() const;
  const AlignmentQualityReport& getAlignmentQualityReport() const;
  void setSparseAnchorUsage(bool enable);
  void setSparseAnchorCorrespondences(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& source_points,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& target_points,
    const std::vector<double>& weights = std::vector<double>(),
    const std::vector<double>& sigmas = std::vector<double>());
  void clearSparseAnchorCorrespondences();

  const Matrix6& getFinalHessian() const;

  double evaluateCost(const Eigen::Matrix4f& relative_pose, Matrix6* H = nullptr, Vector6* b = nullptr);

  virtual void swapSourceAndTarget() {}
  virtual void clearSource() {}
  virtual void clearTarget() {}

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  bool is_converged(const Eigen::Isometry3d& delta) const;

  virtual double linearize(const Eigen::Isometry3d& trans, Matrix6* H = nullptr, Vector6* b = nullptr) = 0;
  virtual double compute_error(const Eigen::Isometry3d& trans) = 0;

  bool step_optimize(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);
  bool step_gn(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);
  bool step_lm(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);

  struct PreparedLinearSystem {
    Matrix6 raw_hessian = Matrix6::Identity();
    Matrix6 regularized_hessian = Matrix6::Identity();
    Vector6 gradient = Vector6::Zero();
    Vector6 eigenvalues = Vector6::Zero();
    Vector6 ambiguity_scores = Vector6::Zero();
    std::array<int, kLsqDof> auto_suppressed_mask{{0, 0, 0, 0, 0, 0}};
    int estimated_rank = kLsqDof;
    double condition_number = 1.0;
    double cost = 0.0;
  };

  PreparedLinearSystem build_linearized_system(const Eigen::Isometry3d& trans, bool force_observability_analysis = false);
  Vector6 solve_linearized_system(const Matrix6& H, const Vector6& b, const std::array<int, kLsqDof>& hard_lock_mask) const;
  double sparse_anchor_cost(const Eigen::Isometry3d& trans, Matrix6* H = nullptr, Vector6* b = nullptr) const;
  void store_observability_diagnostics(const PreparedLinearSystem& system);
  Eigen::Isometry3d apply_hard_locks_to_pose(const Eigen::Isometry3d& previous_pose, const Eigen::Isometry3d& candidate_pose) const;
  void fill_alignment_quality_report(const PreparedLinearSystem& system, const Eigen::Isometry3d& final_pose);
  void collect_anchor_quality_metrics(AlignmentQualityReport* report, const Eigen::Isometry3d& final_pose) const;
  void evaluate_quality_gating(AlignmentQualityReport* report) const;
  virtual void collect_alignment_quality_metrics(AlignmentQualityReport* report, const Eigen::Isometry3d& final_pose) const;

protected:
  double rotation_epsilon_;

  LSQ_OPTIMIZER_TYPE lsq_optimizer_type_;
  int lm_max_iterations_;
  double lm_init_lambda_factor_;
  double lm_lambda_;
  bool lm_debug_print_;

  Matrix6 final_hessian_;
  Matrix6 final_regularized_hessian_;

  ObservabilityConfig observability_config_;
  ObservabilityDiagnostics observability_diagnostics_;
  AlignmentQualityConfig alignment_quality_config_;
  AlignmentQualityReport alignment_quality_report_;

  bool use_sparse_anchors_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> sparse_anchor_source_points_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> sparse_anchor_target_points_;
  std::vector<double> sparse_anchor_weights_;
  std::vector<double> sparse_anchor_sigmas_;
};
}  // namespace fast_gicp

#endif
