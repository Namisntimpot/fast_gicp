// Lightweight self-test for dynamic_rejection helpers. Returns 0 on success,
// non-zero on failure. No gtest dependency.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include <fast_gicp/gicp/dynamic_rejection.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace dyn = fast_gicp::dyn;
using fast_gicp::DynamicRejectionConfig;
using fast_gicp::DynamicRejectionKernel;

namespace {

#define REQUIRE(cond) do { if (!(cond)) { \
  std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
  return 1; } } while (0)

#define APPROX(a, b, eps) do { if (std::abs((a) - (b)) > (eps)) { \
  std::fprintf(stderr, "FAIL %s:%d: %g != %g (tol %g)\n", __FILE__, __LINE__, \
    static_cast<double>(a), static_cast<double>(b), static_cast<double>(eps)); \
  return 1; } } while (0)

int test_mad_basic() {
  std::vector<double> v = {1, 2, 3, 4, 5};  // median 3, |d| {2,1,0,1,2}, median 1
  APPROX(dyn::mad_of_residuals(v), 1.0, 1e-12);
  std::vector<double> empty;
  APPROX(dyn::mad_of_residuals(empty), 0.0, 1e-12);
  std::vector<double> with_nan = {1.0, std::nan(""), 2.0, 3.0, -1.0, 4.0, 5.0};
  // After filter -> {1,2,3,4,5}, MAD = 1
  APPROX(dyn::mad_of_residuals(with_nan), 1.0, 1e-12);
  return 0;
}

int test_geman_mcclure_monotone() {
  // weight is monotonically decreasing in r^2 for fixed mu > 0.
  const double mu = 4.0;
  double prev = dyn::geman_mcclure_weight(0.0, mu);
  REQUIRE(prev == 1.0);
  for (double r2 = 0.5; r2 < 50.0; r2 += 0.5) {
    const double w = dyn::geman_mcclure_weight(r2, mu);
    REQUIRE(w >= 0.0 && w <= 1.0);
    REQUIRE(w < prev);
    prev = w;
  }
  return 0;
}

int test_cauchy_kernel() {
  APPROX(dyn::cauchy_weight(0.0, 1.0), 1.0, 1e-12);
  APPROX(dyn::cauchy_weight(1.0, 1.0), 0.5, 1e-12);
  APPROX(dyn::cauchy_weight(3.0, 1.0), 0.25, 1e-12);
  return 0;
}

int test_compute_weights_with_floor() {
  // 8 residuals: 5 small (inliers), 3 huge (outliers).
  std::vector<double> r = {0.01, 0.02, 0.03, 0.015, 0.025, 10.0, 12.0, 15.0};
  std::vector<double> w;
  // mu chosen comfortably above inlier residuals so inliers keep w >> 0,
  // outliers are deeply suppressed.
  const double mu = 1.0;
  dyn::compute_weights(DynamicRejectionKernel::GEMAN_MCCLURE, mu, r, &w);
  REQUIRE(w.size() == r.size());
  // Each inlier weight must be much larger than each outlier weight.
  double min_inlier = 1.0;
  double max_outlier = 0.0;
  for (size_t i = 0; i < 5; ++i) min_inlier = std::min(min_inlier, w[i]);
  for (size_t i = 5; i < 8; ++i) max_outlier = std::max(max_outlier, w[i]);
  REQUIRE(min_inlier > 0.9);
  REQUIRE(max_outlier < 0.05);
  // Sanity: weights are in [0,1].
  for (double x : w) REQUIRE(x >= 0.0 && x <= 1.0);

  // Now with a 0.9 floor: at least 7 must be inliers, even though kernel only
  // marked 5 by itself. Set mu small so without floor only inliers pass.
  std::vector<double> w_floor;
  const double floor_val = dyn::compute_weights_with_floor(
    DynamicRejectionKernel::GEMAN_MCCLURE, 0.001, r, 0.9, &w_floor);
  APPROX(floor_val, 0.5, 1e-12);
  int n_at_least_half = 0;
  for (double x : w_floor) if (x >= 0.5) ++n_at_least_half;
  REQUIRE(n_at_least_half >= 7);
  return 0;
}

int test_mu_schedule_monotone() {
  DynamicRejectionConfig cfg;
  cfg.gnc_mu_decay = 0.5;
  std::vector<double> r = {0.1, 0.2, 0.3, 5.0, 10.0};
  double mu = dyn::initial_mu_from_residuals(r, cfg);
  double floor_mu = dyn::floor_mu_from_residuals(r, cfg);
  REQUIRE(mu > floor_mu);
  double prev = mu;
  for (int k = 0; k < 20; ++k) {
    const double next = dyn::next_mu(prev, floor_mu, cfg.gnc_mu_decay);
    REQUIRE(next <= prev + 1e-12);
    REQUIRE(next >= floor_mu - 1e-12);
    prev = next;
  }
  APPROX(prev, floor_mu, 1e-9);
  return 0;
}

int test_summarise_weights() {
  std::vector<double> r = {0.1, 0.2, 5.0, 6.0};
  std::vector<double> w = {1.0, 0.8, 0.1, 0.0};
  auto s = dyn::summarise_weights(r, w);
  REQUIRE(s.total == 4);
  REQUIRE(s.inliers == 2);
  APPROX(s.mean_inlier_residual, 0.15, 1e-9);
  APPROX(s.mean_outlier_residual, 5.5, 1e-9);
  return 0;
}

}  // namespace

int main() {
  int rc = 0;
  rc |= test_mad_basic();
  rc |= test_geman_mcclure_monotone();
  rc |= test_cauchy_kernel();
  rc |= test_compute_weights_with_floor();
  rc |= test_mu_schedule_monotone();
  rc |= test_summarise_weights();
  if (rc == 0) std::printf("dynamic_rejection_test: ALL PASS\n");
  return rc;
}
