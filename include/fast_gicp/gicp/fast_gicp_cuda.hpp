#ifndef FAST_GICP_FAST_GICP_CUDA_HPP
#define FAST_GICP_FAST_GICP_CUDA_HPP

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include <fast_gicp/gicp/lsq_registration.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

namespace cuda {
class FastGICPCudaCore;
}

/**
 * @brief CUDA-accelerated GICP that mirrors the three CPU paths actually used:
 *        classic point-to-point GICP, 2DGS surfel covariances, and sparse 3D
 *        anchors. Dynamic outlier rejection is supported.
 *
 * Parallels FastGICP (not derived) so its CPU-only KD-tree codepath stays
 * unchanged.
 */
template<typename PointSource, typename PointTarget>
class FastGICPCuda : public LsqRegistration<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  using Ptr = pcl::shared_ptr<FastGICPCuda<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const FastGICPCuda<PointSource, PointTarget>>;
#else
  using Ptr = boost::shared_ptr<FastGICPCuda<PointSource, PointTarget>>;
  using ConstPtr = boost::shared_ptr<const FastGICPCuda<PointSource, PointTarget>>;
#endif

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;

public:
  FastGICPCuda();
  virtual ~FastGICPCuda() override;

  void setCorrespondenceRandomness(int k);
  void setRegularizationMethod(RegularizationMethod method);

  // KNN backend selection. "brute_force" is the default (uses
  // brute_force_knn.cu already in the repo). "cuvs" requires USE_CUVS=ON at
  // build time and the cuVS shared library at runtime; throws otherwise.
  void setKnnBackend(const std::string& backend);
  const std::string& getKnnBackend() const;

  virtual void swapSourceAndTarget() override;
  virtual void clearSource() override;
  virtual void clearTarget() override;

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

  // 2DGS surfel covariance entry point (mirrors FastGICP).
  void setSourceCovariances2DGS(
    const std::vector<float>& rotationsq_xyzw,
    const std::vector<float>& scales_2d,
    const std::string& mode,
    double normal_sigma_ratio,
    double normal_sigma_min);
  void setTargetCovariances2DGS(
    const std::vector<float>& rotationsq_xyzw,
    const std::vector<float>& scales_2d,
    const std::string& mode,
    double normal_sigma_ratio,
    double normal_sigma_min);

  // Diagnostic accessors. Once kernels are filled in B2/B4 these mirror the
  // CPU FastGICP getters.
  int getSourceSize() const;
  int getTargetSize() const;

  // Per-source-point correspondence index and squared distance after the
  // latest align() call. Mirrors FastGICP's getSourceCorrespondences /
  // getSourceSqDistances so frontend code that probes match quality (e.g.
  // matched_ratio, keyframe_ratio) works unchanged on the CUDA path. Each
  // call refreshes the host-side cache via a device->host copy.
  const std::vector<int>& getSourceCorrespondences() const;
  const std::vector<float>& getSourceSqDistances() const;

protected:
  virtual bool supports_dynamic_rejection() const override { return true; }

  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;
  virtual double linearize(const Eigen::Isometry3d& trans,
                           Eigen::Matrix<double, 6, 6>* H = nullptr,
                           Eigen::Matrix<double, 6, 1>* b = nullptr) override;
  virtual double compute_error(const Eigen::Isometry3d& trans) override;
  virtual int current_geometric_term_count() const override;
  virtual void collect_alignment_quality_metrics(AlignmentQualityReport* report,
                                                 const Eigen::Isometry3d& final_pose) const override;

private:
  int k_correspondences_;
  RegularizationMethod regularization_method_;
  std::string knn_backend_;

  std::unique_ptr<cuda::FastGICPCudaCore> impl_;

  // Host-side caches for the diagnostic accessors above. Mutable because the
  // getters are const but lazily refresh from the device.
  mutable std::vector<int> correspondences_host_;
  mutable std::vector<float> sq_distances_host_;
};

}  // namespace fast_gicp

#endif
