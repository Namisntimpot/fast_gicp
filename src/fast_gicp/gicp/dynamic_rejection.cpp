#include <fast_gicp/gicp/dynamic_rejection.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace fast_gicp {
namespace dyn {

namespace {

double median_inplace(std::vector<double>* values) {
  if (values->empty()) {
    return 0.0;
  }
  const std::size_t n = values->size();
  const std::size_t mid = n / 2;
  std::nth_element(values->begin(), values->begin() + mid, values->end());
  const double upper = (*values)[mid];
  if (n % 2 == 1) {
    return upper;
  }
  // even count: average of the two middle elements; nth_element guarantees
  // everything to the left is <= upper.
  const double lower = *std::max_element(values->begin(), values->begin() + mid);
  return 0.5 * (lower + upper);
}

}  // namespace

double mad_of_residuals_into(const std::vector<double>& residuals, std::vector<double>* scratch) {
  scratch->clear();
  scratch->reserve(residuals.size());
  for (double r : residuals) {
    if (std::isfinite(r) && r >= 0.0) {
      scratch->push_back(r);
    }
  }
  if (scratch->empty()) {
    return 0.0;
  }
  const double med = median_inplace(scratch);
  // re-use buffer for |r - median|
  for (double& r : *scratch) {
    r = std::abs(r - med);
  }
  return median_inplace(scratch);
}

double mad_of_residuals(const std::vector<double>& residuals) {
  std::vector<double> scratch;
  return mad_of_residuals_into(residuals, &scratch);
}

double geman_mcclure_weight(double r2, double mu) {
  if (mu <= 0.0) {
    return r2 <= 0.0 ? 1.0 : 0.0;
  }
  const double ratio = mu / (mu + r2);
  return ratio * ratio;
}

double cauchy_weight(double r2, double mu) {
  if (mu <= 0.0) {
    return r2 <= 0.0 ? 1.0 : 0.0;
  }
  return 1.0 / (1.0 + r2 / mu);
}

double kernel_weight(DynamicRejectionKernel kernel, double r2, double mu) {
  if (!std::isfinite(r2) || r2 < 0.0) {
    return 0.0;
  }
  switch (kernel) {
    case DynamicRejectionKernel::NONE:
      return 1.0;
    case DynamicRejectionKernel::GEMAN_MCCLURE:
      return geman_mcclure_weight(r2, mu);
    case DynamicRejectionKernel::CAUCHY:
      return cauchy_weight(r2, mu);
  }
  return 1.0;
}

void compute_weights(
  DynamicRejectionKernel kernel,
  double mu,
  const std::vector<double>& residuals,
  std::vector<double>* weights) {
  weights->resize(residuals.size());
  for (std::size_t i = 0; i < residuals.size(); ++i) {
    (*weights)[i] = kernel_weight(kernel, residuals[i], mu);
  }
}

double compute_weights_with_floor(
  DynamicRejectionKernel kernel,
  double mu,
  const std::vector<double>& residuals,
  double min_inlier_ratio,
  std::vector<double>* weights) {
  compute_weights(kernel, mu, residuals, weights);
  if (min_inlier_ratio <= 0.0 || weights->empty()) {
    return 0.0;
  }

  // Count how many points are already inliers (w > 0.5).
  std::size_t inliers = 0;
  for (double w : *weights) {
    if (w > 0.5) ++inliers;
  }
  const std::size_t required = static_cast<std::size_t>(std::ceil(min_inlier_ratio * weights->size()));
  if (inliers >= required) {
    return 0.0;
  }

  // We need to raise some weights to 0.5+ so that at least `required` are inliers.
  // Pick the smallest residuals (most likely the closest to being inliers) and
  // set their weight to 0.5. This is an emergency floor: it limits how aggressive
  // GNC can be.
  std::vector<std::size_t> idx(residuals.size());
  for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + required, idx.end(),
    [&residuals](std::size_t a, std::size_t b) {
      return residuals[a] < residuals[b];
    });
  for (std::size_t k = 0; k < required && k < idx.size(); ++k) {
    if ((*weights)[idx[k]] < 0.5) {
      (*weights)[idx[k]] = 0.5;
    }
  }
  return 0.5;
}

double initial_mu_from_residuals(
  const std::vector<double>& residuals,
  const DynamicRejectionConfig& config) {
  if (residuals.empty()) return 1.0;
  const double mad = mad_of_residuals(residuals);
  const double sigma = std::max(mad * config.mad_scale, 1e-12);
  return sigma * sigma * std::max(config.gnc_mu_init_scale, 1e-3);
}

double floor_mu_from_residuals(
  const std::vector<double>& residuals,
  const DynamicRejectionConfig& config) {
  if (residuals.empty()) return 1e-12;
  const double mad = mad_of_residuals(residuals);
  const double sigma = std::max(mad * config.mad_scale, 1e-12);
  return sigma * sigma * std::max(config.gnc_mu_floor_scale, 1e-6);
}

double next_mu(double current_mu, double floor_mu, double decay) {
  const double safe_decay = std::min(1.0, std::max(1e-6, decay));
  return std::max(current_mu * safe_decay, floor_mu);
}

WeightStats summarise_weights(
  const std::vector<double>& residuals,
  const std::vector<double>& weights) {
  WeightStats s;
  s.total = static_cast<int>(weights.size());
  double inlier_sum = 0.0;
  double outlier_sum = 0.0;
  int outliers = 0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    const double r = (i < residuals.size() && std::isfinite(residuals[i])) ? residuals[i] : 0.0;
    if (weights[i] >= 0.5) {
      ++s.inliers;
      inlier_sum += r;
    } else {
      ++outliers;
      outlier_sum += r;
    }
  }
  s.mean_inlier_residual = (s.inliers > 0) ? (inlier_sum / s.inliers) : 0.0;
  s.mean_outlier_residual = (outliers > 0) ? (outlier_sum / outliers) : 0.0;
  return s;
}

}  // namespace dyn
}  // namespace fast_gicp
