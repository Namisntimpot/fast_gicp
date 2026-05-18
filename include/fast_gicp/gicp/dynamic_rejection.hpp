#ifndef FAST_GICP_DYNAMIC_REJECTION_HPP
#define FAST_GICP_DYNAMIC_REJECTION_HPP

#include <vector>
#include <cstddef>

#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {
namespace dyn {

// Median absolute deviation of a set of non-negative residuals (e.g. squared
// Mahalanobis distances). NaN-safe; values < 0 ignored. Returns 0 if empty.
// The input vector is NOT modified.
double mad_of_residuals(const std::vector<double>& residuals);

// Same as above, but operates on a contiguous slice. The buffer 'scratch' is
// used for nth_element so the caller's data remains intact. Non-allocating
// when scratch is already sized.
double mad_of_residuals_into(const std::vector<double>& residuals, std::vector<double>* scratch);

// Geman-McClure weight for residual squared r2 under shape mu:
//   w(r) = (mu / (mu + r^2))^2, clamped to [0, 1].
double geman_mcclure_weight(double r2, double mu);

// Cauchy weight: w(r) = 1 / (1 + r^2 / c^2). With c^2 = mu for parameterisation
// uniformity with GNC schedules.
double cauchy_weight(double r2, double mu);

// Dispatch to the configured kernel. r2 must be >= 0; returns 1.0 if kernel == NONE.
double kernel_weight(DynamicRejectionKernel kernel, double r2, double mu);

// Apply the kernel to every residual. weights is resized to residuals.size().
void compute_weights(
  DynamicRejectionKernel kernel,
  double mu,
  const std::vector<double>& residuals,
  std::vector<double>* weights);

// Same as compute_weights but enforces min_inlier_ratio by raising the smallest
// weights to a floor that preserves the requested ratio of >= 0.5 weights.
// Returns the floor used (0 if no clamping needed).
double compute_weights_with_floor(
  DynamicRejectionKernel kernel,
  double mu,
  const std::vector<double>& residuals,
  double min_inlier_ratio,
  std::vector<double>* weights);

// Initial mu given MAD of residuals and config. Uses (mad_scale * MAD)^2 *
// gnc_mu_init_scale, with a floor at 1e-12 to keep arithmetic well-defined.
double initial_mu_from_residuals(
  const std::vector<double>& residuals,
  const DynamicRejectionConfig& config);

// Floor mu (lower bound) given the same residuals/config.
double floor_mu_from_residuals(
  const std::vector<double>& residuals,
  const DynamicRejectionConfig& config);

// One GNC schedule step: mu_next = max(mu * decay, floor_mu).
double next_mu(double current_mu, double floor_mu, double decay);

// Compute inlier count (weight > 0.5) and accumulated residual statistics.
struct WeightStats {
  int total = 0;
  int inliers = 0;
  double mean_inlier_residual = 0.0;
  double mean_outlier_residual = 0.0;
};
WeightStats summarise_weights(
  const std::vector<double>& residuals,
  const std::vector<double>& weights);

}  // namespace dyn
}  // namespace fast_gicp

#endif
