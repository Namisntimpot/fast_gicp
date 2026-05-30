#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <boost/filesystem.hpp>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/print.h>
#include <pcl/filters/approximate_voxel_grid.h>

namespace py = pybind11;

namespace {

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> numpy_to_vec3_list(const py::array& array) {
  py::array_t<double, py::array::c_style | py::array::forcecast> casted(array);
  py::buffer_info info = casted.request();

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> values;
  if (info.ndim == 2 && info.shape[1] == 3) {
    values.resize(info.shape[0]);
    auto* data = static_cast<double*>(info.ptr);
    for (ssize_t i = 0; i < info.shape[0]; i++) {
      values[i] = Eigen::Vector3d(data[3 * i + 0], data[3 * i + 1], data[3 * i + 2]);
    }
    return values;
  }

  if (info.ndim == 1 && info.shape[0] % 3 == 0) {
    values.resize(info.shape[0] / 3);
    auto* data = static_cast<double*>(info.ptr);
    for (ssize_t i = 0; i < values.size(); i++) {
      values[i] = Eigen::Vector3d(data[3 * i + 0], data[3 * i + 1], data[3 * i + 2]);
    }
    return values;
  }

  throw std::invalid_argument("expected an Nx3 or flat 3N numpy array");
}

std::array<int, fast_gicp::kLsqDof> numpy_to_dof_mask(const py::array& array) {
  py::array_t<int, py::array::c_style | py::array::forcecast> casted(array);
  py::buffer_info info = casted.request();
  if (info.ndim != 1 || info.shape[0] != fast_gicp::kLsqDof) {
    throw std::invalid_argument("expected a flat length-6 mask");
  }

  std::array<int, fast_gicp::kLsqDof> mask{{0, 0, 0, 0, 0, 0}};
  auto* data = static_cast<int*>(info.ptr);
  for (int i = 0; i < fast_gicp::kLsqDof; i++) {
    mask[i] = data[i] ? 1 : 0;
  }
  return mask;
}

py::array_t<int> dof_mask_to_numpy(const Eigen::Matrix<int, fast_gicp::kLsqDof, 1>& mask) {
  py::array_t<int> result({fast_gicp::kLsqDof});
  auto view = result.mutable_unchecked<1>();
  for (int i = 0; i < fast_gicp::kLsqDof; i++) {
    view(i) = mask[i];
  }
  return result;
}

py::array_t<double> dof_scores_to_numpy(const Eigen::Matrix<double, fast_gicp::kLsqDof, 1>& scores) {
  py::array_t<double> result({fast_gicp::kLsqDof});
  auto view = result.mutable_unchecked<1>();
  for (int i = 0; i < fast_gicp::kLsqDof; i++) {
    view(i) = scores[i];
  }
  return result;
}

py::array_t<double> dof_thresholds_to_numpy(const std::array<double, fast_gicp::kLsqDof>& thresholds) {
  py::array_t<double> result({fast_gicp::kLsqDof});
  auto view = result.mutable_unchecked<1>();
  for (int i = 0; i < fast_gicp::kLsqDof; i++) {
    view(i) = thresholds[i];
  }
  return result;
}

std::vector<double> numpy_to_double_list(const py::array& array) {
  py::array_t<double, py::array::c_style | py::array::forcecast> casted(array);
  py::buffer_info info = casted.request();
  if (info.ndim != 1) {
    throw std::invalid_argument("expected a flat numpy array");
  }
  auto* data = static_cast<double*>(info.ptr);
  return std::vector<double>(data, data + info.shape[0]);
}

std::string optimizer_name(fast_gicp::LSQ_OPTIMIZER_TYPE type) {
  switch (type) {
    case fast_gicp::LSQ_OPTIMIZER_TYPE::GaussNewton:
      return "GaussNewton";
    case fast_gicp::LSQ_OPTIMIZER_TYPE::LevenbergMarquardt:
      return "LevenbergMarquardt";
  }
  return "LevenbergMarquardt";
}

fast_gicp::LSQ_OPTIMIZER_TYPE optimizer_type(const std::string& name) {
  if (name == "GaussNewton") {
    return fast_gicp::LSQ_OPTIMIZER_TYPE::GaussNewton;
  }
  if (name == "LevenbergMarquardt") {
    return fast_gicp::LSQ_OPTIMIZER_TYPE::LevenbergMarquardt;
  }
  throw std::invalid_argument("unknown optimizer type: " + name);
}

fast_gicp::ColorMatchingConfig color_matching_config_from_args(bool enabled, int candidate_count, double color_weight, double color_sigma) {
  fast_gicp::ColorMatchingConfig config;
  config.enable_color_matching = enabled;
  config.geometric_candidate_count = candidate_count;
  config.color_weight = color_weight;
  config.color_sigma = color_sigma;
  return config;
}

std::string sparse_anchor_balance_mode_name(fast_gicp::SparseAnchorBalanceMode mode) {
  switch (mode) {
    case fast_gicp::SparseAnchorBalanceMode::NONE:
      return "NONE";
    case fast_gicp::SparseAnchorBalanceMode::BY_COUNT:
      return "BY_COUNT";
    case fast_gicp::SparseAnchorBalanceMode::BY_HESSIAN_TRACE:
      return "BY_HESSIAN_TRACE";
  }

  return "NONE";
}

fast_gicp::SparseAnchorBalanceMode sparse_anchor_balance_mode(const py::handle& value) {
  if (py::isinstance<py::str>(value)) {
    const std::string mode = py::cast<std::string>(value);
    if (mode == "NONE") {
      return fast_gicp::SparseAnchorBalanceMode::NONE;
    }
    if (mode == "BY_COUNT") {
      return fast_gicp::SparseAnchorBalanceMode::BY_COUNT;
    }
    if (mode == "BY_HESSIAN_TRACE") {
      return fast_gicp::SparseAnchorBalanceMode::BY_HESSIAN_TRACE;
    }
    throw std::invalid_argument("unknown sparse anchor balance mode: " + mode);
  }

  const int mode = py::cast<int>(value);
  switch (mode) {
    case 0:
      return fast_gicp::SparseAnchorBalanceMode::NONE;
    case 1:
      return fast_gicp::SparseAnchorBalanceMode::BY_COUNT;
    case 2:
      return fast_gicp::SparseAnchorBalanceMode::BY_HESSIAN_TRACE;
    default:
      throw std::invalid_argument("unknown sparse anchor balance mode index");
  }
}

std::array<double, fast_gicp::kLsqDof> sequence_to_dof_thresholds(const py::handle& value) {
  py::sequence seq = value.cast<py::sequence>();
  if (py::len(seq) != fast_gicp::kLsqDof) {
    throw std::invalid_argument("expected a length-6 sequence");
  }

  std::array<double, fast_gicp::kLsqDof> thresholds{};
  for (int i = 0; i < fast_gicp::kLsqDof; i++) {
    thresholds[i] = py::cast<double>(seq[i]);
  }
  return thresholds;
}

py::dict alignment_quality_config_to_dict(const fast_gicp::AlignmentQualityConfig& config) {
  py::dict info;
  info["enable_suggested_gating"] = config.enable_suggested_gating;
  info["require_converged"] = config.require_converged;
  info["min_correspondence_count"] = config.min_correspondence_count;
  info["min_matched_count"] = config.min_matched_count;
  info["min_matched_ratio"] = config.min_matched_ratio;
  info["max_fitness_score"] = config.max_fitness_score;
  info["max_final_cost"] = config.max_final_cost;
  info["max_normalized_cost_per_match"] = config.max_normalized_cost_per_match;
  info["max_mean_sq_distance"] = config.max_mean_sq_distance;
  info["max_median_sq_distance"] = config.max_median_sq_distance;
  info["max_p90_sq_distance"] = config.max_p90_sq_distance;
  info["max_p95_sq_distance"] = config.max_p95_sq_distance;
  info["min_rank"] = config.min_rank;
  info["max_condition_number"] = config.max_condition_number;
  info["max_ambiguity"] = config.max_ambiguity;
  info["max_dof_ambiguity"] = dof_thresholds_to_numpy(config.max_dof_ambiguity);
  info["max_anchor_mean_residual"] = config.max_anchor_mean_residual;
  info["max_anchor_p95_residual"] = config.max_anchor_p95_residual;
  return info;
}

py::dict sparse_anchor_config_to_dict(const fast_gicp::SparseAnchorConfig& config) {
  py::dict info;
  info["objective_weight"] = config.objective_weight;
  info["balance_mode"] = sparse_anchor_balance_mode_name(config.balance_mode);
  info["auto_balance_min"] = config.auto_balance_min;
  info["auto_balance_max"] = config.auto_balance_max;
  return info;
}

void update_sparse_anchor_config_from_dict(
  fast_gicp::SparseAnchorConfig* config,
  const py::dict& options) {
  for (auto item : options) {
    const std::string key = py::cast<std::string>(item.first);
    const py::handle value = item.second;

    if (key == "objective_weight") {
      config->objective_weight = py::cast<double>(value);
    } else if (key == "balance_mode") {
      config->balance_mode = sparse_anchor_balance_mode(value);
    } else if (key == "auto_balance_min") {
      config->auto_balance_min = py::cast<double>(value);
    } else if (key == "auto_balance_max") {
      config->auto_balance_max = py::cast<double>(value);
    } else {
      throw std::invalid_argument("unknown sparse anchor config key: " + key);
    }
  }
}

void update_alignment_quality_config_from_dict(
  fast_gicp::AlignmentQualityConfig* config,
  const py::dict& options) {
  for (auto item : options) {
    const std::string key = py::cast<std::string>(item.first);
    const py::handle value = item.second;

    if (key == "enable_suggested_gating") {
      config->enable_suggested_gating = py::cast<bool>(value);
    } else if (key == "require_converged") {
      config->require_converged = py::cast<bool>(value);
    } else if (key == "min_correspondence_count") {
      config->min_correspondence_count = py::cast<int>(value);
    } else if (key == "min_matched_count") {
      config->min_matched_count = py::cast<int>(value);
    } else if (key == "min_matched_ratio") {
      config->min_matched_ratio = py::cast<double>(value);
    } else if (key == "max_fitness_score") {
      config->max_fitness_score = py::cast<double>(value);
    } else if (key == "max_final_cost") {
      config->max_final_cost = py::cast<double>(value);
    } else if (key == "max_normalized_cost_per_match") {
      config->max_normalized_cost_per_match = py::cast<double>(value);
    } else if (key == "max_mean_sq_distance") {
      config->max_mean_sq_distance = py::cast<double>(value);
    } else if (key == "max_median_sq_distance") {
      config->max_median_sq_distance = py::cast<double>(value);
    } else if (key == "max_p90_sq_distance") {
      config->max_p90_sq_distance = py::cast<double>(value);
    } else if (key == "max_p95_sq_distance") {
      config->max_p95_sq_distance = py::cast<double>(value);
    } else if (key == "min_rank") {
      config->min_rank = py::cast<int>(value);
    } else if (key == "max_condition_number") {
      config->max_condition_number = py::cast<double>(value);
    } else if (key == "max_ambiguity") {
      config->max_ambiguity = py::cast<double>(value);
    } else if (key == "max_dof_ambiguity") {
      config->max_dof_ambiguity = sequence_to_dof_thresholds(value);
    } else if (key == "max_anchor_mean_residual") {
      config->max_anchor_mean_residual = py::cast<double>(value);
    } else if (key == "max_anchor_p95_residual") {
      config->max_anchor_p95_residual = py::cast<double>(value);
    } else {
      throw std::invalid_argument("unknown alignment quality config key: " + key);
    }
  }
}

py::dict alignment_quality_report_to_dict(const fast_gicp::AlignmentQualityReport& report) {
  py::dict info;
  info["valid"] = report.valid;
  info["converged"] = report.converged;
  info["gating_evaluated"] = report.gating_evaluated;
  info["suggested_accept"] = report.suggested_accept;
  info["has_match_statistics"] = report.has_match_statistics;
  info["has_anchor_statistics"] = report.has_anchor_statistics;
  info["used_sparse_anchors"] = report.used_sparse_anchors;
  info["used_color_matching"] = report.used_color_matching;
  info["optimizer_type"] = report.optimizer_type;
  info["num_iterations"] = report.num_iterations;
  info["source_count"] = report.source_count;
  info["target_count"] = report.target_count;
  info["correspondence_count"] = report.correspondence_count;
  info["matched_count"] = report.matched_count;
  info["anchor_count"] = report.anchor_count;
  info["rank"] = report.rank;
  info["fitness_score"] = report.fitness_score;
  info["final_cost"] = report.final_cost;
  info["normalized_cost_per_match"] = report.normalized_cost_per_match;
  info["matched_ratio"] = report.matched_ratio;
  info["mean_sq_distance"] = report.mean_sq_distance;
  info["median_sq_distance"] = report.median_sq_distance;
  info["p90_sq_distance"] = report.p90_sq_distance;
  info["p95_sq_distance"] = report.p95_sq_distance;
  info["condition_number"] = report.condition_number;
  info["max_ambiguity"] = report.max_ambiguity;
  info["anchor_mean_residual"] = report.anchor_mean_residual;
  info["anchor_p95_residual"] = report.anchor_p95_residual;
  info["anchor_objective_weight"] = report.anchor_objective_weight;
  info["anchor_balance_mode"] = sparse_anchor_balance_mode_name(report.anchor_balance_mode);
  info["anchor_auto_balance_factor"] = report.anchor_auto_balance_factor;
  info["anchor_effective_scale"] = report.anchor_effective_scale;
  info["geometry_raw_cost"] = report.geometry_raw_cost;
  info["anchor_raw_cost"] = report.anchor_raw_cost;
  info["anchor_scaled_cost"] = report.anchor_scaled_cost;
  info["geometry_hessian_trace"] = report.geometry_hessian_trace;
  info["anchor_hessian_trace"] = report.anchor_hessian_trace;
  info["anchor_balance_fallback_used"] = report.anchor_balance_fallback_used;
  info["ambiguity_scores"] = dof_scores_to_numpy(report.ambiguity_scores);
  info["auto_suppressed_mask"] = dof_mask_to_numpy(report.auto_suppressed_mask);
  info["hard_lock_mask"] = dof_mask_to_numpy(report.hard_lock_mask);
  info["smooth_regularization_weights"] = dof_scores_to_numpy(report.smooth_regularization_weights);
  info["geometry_rank"] = report.geometry_rank;
  info["geometry_condition_number"] = report.geometry_condition_number;
  info["rejection_reasons"] = report.rejection_reasons;
  return info;
}

}  // namespace

fast_gicp::NeighborSearchMethod search_method(const std::string& neighbor_search_method) {
  if(neighbor_search_method == "DIRECT1") {
    return fast_gicp::NeighborSearchMethod::DIRECT1;
  } else if (neighbor_search_method == "DIRECT7") {
    return fast_gicp::NeighborSearchMethod::DIRECT7;
  } else if (neighbor_search_method == "DIRECT27") {
    return fast_gicp::NeighborSearchMethod::DIRECT27;
  } else if (neighbor_search_method == "DIRECT_RADIUS") {
    return fast_gicp::NeighborSearchMethod::DIRECT_RADIUS;
  }

  std::cerr << "error: unknown neighbor search method " << neighbor_search_method << std::endl;
  return fast_gicp::NeighborSearchMethod::DIRECT1;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr eigen2pcl(const Eigen::Matrix<double, -1, 3>& points) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->resize(points.rows());

  for(int i=0; i<points.rows(); i++) {
    cloud->at(i).getVector3fMap() = points.row(i).cast<float>();
  }
  return cloud;
}

Eigen::Matrix<double, -1, 3> downsample(const Eigen::Matrix<double, -1, 3>& points, double downsample_resolution) {
  auto cloud = eigen2pcl(points);

  pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
  voxelgrid.setInputCloud(cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  voxelgrid.filter(*filtered);

  Eigen::Matrix<float, -1, 3> filtered_points(filtered->size(), 3);
  for(int i=0; i<filtered->size(); i++) {
    filtered_points.row(i) = filtered->at(i).getVector3fMap();
  }

  return filtered_points.cast<double>();
}

Eigen::Matrix4d align_points(
  const Eigen::Matrix<double, -1, 3>& target,
  const Eigen::Matrix<double, -1, 3>& source,
  const std::string& method,
  double downsample_resolution,
  int k_correspondences,
  double max_correspondence_distance,
  double voxel_resolution,
  int num_threads,
  const std::string& neighbor_search_method,
  double neighbor_search_radius,
  const Eigen::Matrix4f& initial_guess
) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = eigen2pcl(target);
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = eigen2pcl(source);

  if(downsample_resolution > 0.0) {
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    voxelgrid.setInputCloud(target_cloud);
    voxelgrid.filter(*filtered);
    target_cloud.swap(filtered);

    voxelgrid.setInputCloud(source_cloud);
    voxelgrid.filter(*filtered);
    source_cloud.swap(filtered);
  }
  std::shared_ptr<fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>> reg;
  if(method == "GICP") {
    std::shared_ptr<fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>> gicp(new fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>);
    gicp->setMaxCorrespondenceDistance(max_correspondence_distance);
    gicp->setCorrespondenceRandomness(k_correspondences);
    gicp->setNumThreads(num_threads);
    reg = gicp;
  } else if (method == "VGICP") {
    std::shared_ptr<fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>> vgicp(new fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>);
    vgicp->setCorrespondenceRandomness(k_correspondences);
    vgicp->setResolution(voxel_resolution);
    vgicp->setNeighborSearchMethod(search_method(neighbor_search_method));
    vgicp->setNumThreads(num_threads);
    reg = vgicp;
  } else if (method == "VGICP_CUDA") {
#ifdef USE_VGICP_CUDA
    std::shared_ptr<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>> vgicp(new fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>);
    vgicp->setCorrespondenceRandomness(k_correspondences);
    vgicp->setNeighborSearchMethod(search_method(neighbor_search_method), neighbor_search_radius);
    vgicp->setResolution(voxel_resolution);
    reg = vgicp;
#else
    std::cerr << "error: you need to build fast_gicp with BUILD_VGICP_CUDA=ON" << std::endl;
    return Eigen::Matrix4d::Identity();
#endif
  } else if (method == "NDT_CUDA") {
#ifdef USE_VGICP_CUDA
    std::shared_ptr<fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>> ndt(new fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>);
    ndt->setResolution(voxel_resolution);
    ndt->setNeighborSearchMethod(search_method(neighbor_search_method), neighbor_search_radius);
    reg = ndt;
#else
    std::cerr << "error: you need to build fast_gicp with BUILD_VGICP_CUDA=ON" << std::endl;
    return Eigen::Matrix4d::Identity();
#endif
  } else {
    std::cerr << "error: unknown registration method " << method << std::endl;
    return Eigen::Matrix4d::Identity();
  }
  reg->setInputTarget(target_cloud);
  reg->setInputSource(source_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
  reg->align(*aligned, initial_guess);
  return reg->getFinalTransformation().cast<double>();
}


using LsqRegistration = fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>;
using FastGICP = fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>;
using FastVGICP = fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>;
#ifdef USE_VGICP_CUDA
using FastVGICPCuda = fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>;
using NDTCuda = fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>;
#endif

PYBIND11_MODULE(pygicp, m) {
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  m.def("downsample", &downsample, "downsample points");

  m.def("align_points", &align_points, "align two point sets",
    py::arg("target"),
    py::arg("source"),
    py::arg("method") = "GICP",
    py::arg("downsample_resolution") = -1.0,
    py::arg("k_correspondences") = 15,
    py::arg("max_correspondence_distance") = std::numeric_limits<double>::max(),
    py::arg("voxel_resolution") = 1.0,
    py::arg("num_threads") = 0,
    py::arg("neighbor_search_method") = "DIRECT1",
    py::arg("neighbor_search_radius") = 1.5,
    py::arg("initial_guess") = Eigen::Matrix4f::Identity()
  );

  py::class_<LsqRegistration, std::shared_ptr<LsqRegistration>>(m, "LsqRegistration")
    .def("set_input_target", [] (LsqRegistration& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputTarget(eigen2pcl(points)); })
    .def("set_input_source", [] (LsqRegistration& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputSource(eigen2pcl(points)); })
    .def("swap_source_and_target", &LsqRegistration::swapSourceAndTarget)
    .def("set_lsq_optimizer_type", [] (LsqRegistration& reg, const std::string& name) { reg.setLSQOptimizerType(optimizer_type(name)); })
    .def("set_observability_check", &LsqRegistration::setObservabilityCheck)
    .def("set_observability_eigen_thresholds", &LsqRegistration::setObservabilityEigenThresholds)
    .def("set_ambiguity_score_threshold", &LsqRegistration::setAmbiguityScoreThreshold)
    .def("set_auto_soft_prior_strength", &LsqRegistration::setAutoSoftPriorStrength)
    .def("set_hard_lock_mask", [] (LsqRegistration& reg, const py::array& mask) { reg.setHardLockMask(numpy_to_dof_mask(mask)); })
    .def("set_preferred_ambiguous_mask", [] (LsqRegistration& reg, const py::array& mask) { reg.setPreferredAmbiguousMask(numpy_to_dof_mask(mask)); })
    .def("set_prefer_user_marked_dofs", &LsqRegistration::setPreferUserMarkedDofs)
    .def("set_enable_observability_diagnostics", &LsqRegistration::setEnableObservabilityDiagnostics)
    // Smooth prior setters
    .def("set_use_smooth_prior", [](LsqRegistration& reg, bool enable) {
      auto config = reg.getObservabilityConfig();
      config.use_smooth_prior = enable;
      reg.setObservabilityConfig(config);
    })
    .def("set_smooth_prior_falloff", [](LsqRegistration& reg, double falloff) {
      auto config = reg.getObservabilityConfig();
      config.smooth_prior_falloff = falloff;
      reg.setObservabilityConfig(config);
    })
    .def("set_smooth_prior_max_strength", [](LsqRegistration& reg, double strength) {
      auto config = reg.getObservabilityConfig();
      config.smooth_prior_max_strength = strength;
      reg.setObservabilityConfig(config);
    })
    .def("set_analyze_geometry_separately", [](LsqRegistration& reg, bool enable) {
      auto config = reg.getObservabilityConfig();
      config.analyze_geometry_separately = enable;
      reg.setObservabilityConfig(config);
    })
    .def("set_anchor_aware_regularization", [](LsqRegistration& reg, bool enable) {
      auto config = reg.getObservabilityConfig();
      config.anchor_aware_regularization = enable;
      reg.setObservabilityConfig(config);
    })
    .def("set_regularization_scale_mode", [](LsqRegistration& reg, int mode) {
      auto config = reg.getObservabilityConfig();
      config.regularization_scale_mode = mode;
      reg.setObservabilityConfig(config);
    })
    .def("set_alignment_quality_config", [] (LsqRegistration& reg, const py::dict& options) {
      auto config = reg.getAlignmentQualityConfig();
      update_alignment_quality_config_from_dict(&config, options);
      reg.setAlignmentQualityConfig(config);
    })
    .def("set_sparse_anchor_config", [] (LsqRegistration& reg, const py::dict& options) {
      auto config = reg.getSparseAnchorConfig();
      update_sparse_anchor_config_from_dict(&config, options);
      reg.setSparseAnchorConfig(config);
    })
    .def("set_sparse_anchor_objective_weight", &LsqRegistration::setSparseAnchorObjectiveWeight)
    .def("set_sparse_anchor_balance_mode", [] (LsqRegistration& reg, const py::object& mode) {
      reg.setSparseAnchorBalanceMode(sparse_anchor_balance_mode(mode));
    })
    .def("set_use_sparse_anchors", &LsqRegistration::setSparseAnchorUsage)
    .def("set_sparse_anchor_correspondences", [] (LsqRegistration& reg, const py::array& source_points, const py::array& target_points, const py::object& weights, const py::object& sigmas) {
      const auto source = numpy_to_vec3_list(source_points);
      const auto target = numpy_to_vec3_list(target_points);
      const auto weight_values = weights.is_none() ? std::vector<double>() : numpy_to_double_list(weights.cast<py::array>());
      const auto sigma_values = sigmas.is_none() ? std::vector<double>() : numpy_to_double_list(sigmas.cast<py::array>());
      reg.setSparseAnchorCorrespondences(source, target, weight_values, sigma_values);
    }, py::arg("source_points"), py::arg("target_points"), py::arg("weights") = py::none(), py::arg("sigmas") = py::none())
    .def("clear_sparse_anchor_correspondences", &LsqRegistration::clearSparseAnchorCorrespondences)
    .def("get_final_hessian", &LsqRegistration::getFinalHessian)
    .def("get_final_regularized_hessian", &LsqRegistration::getFinalRegularizedHessian)
    .def("get_final_transformation", &LsqRegistration::getFinalTransformation)
    .def("get_observability_config", [] (LsqRegistration& reg) {
      const auto& config = reg.getObservabilityConfig();
      py::dict info;
      info["enable_observability_check"] = config.enable_observability_check;
      info["relative_eigenvalue_threshold"] = config.relative_eigenvalue_threshold;
      info["absolute_eigenvalue_threshold"] = config.absolute_eigenvalue_threshold;
      info["ambiguity_score_threshold"] = config.ambiguity_score_threshold;
      info["auto_soft_prior_strength"] = config.auto_soft_prior_strength;
      info["hard_lock_mask"] = py::cast(std::vector<int>(config.hard_lock_mask.begin(), config.hard_lock_mask.end()));
      info["preferred_ambiguous_mask"] = py::cast(std::vector<int>(config.preferred_ambiguous_mask.begin(), config.preferred_ambiguous_mask.end()));
      info["prefer_user_marked_dofs"] = config.prefer_user_marked_dofs;
      info["enable_diagnostics"] = config.enable_diagnostics;
      info["use_smooth_prior"] = config.use_smooth_prior;
      info["smooth_prior_falloff"] = config.smooth_prior_falloff;
      info["smooth_prior_max_strength"] = config.smooth_prior_max_strength;
      info["analyze_geometry_separately"] = config.analyze_geometry_separately;
      info["anchor_aware_regularization"] = config.anchor_aware_regularization;
      info["regularization_scale_mode"] = config.regularization_scale_mode;
      return info;
    })
    .def("get_alignment_quality_config", [] (LsqRegistration& reg) {
      return alignment_quality_config_to_dict(reg.getAlignmentQualityConfig());
    })
    .def("get_sparse_anchor_config", [] (LsqRegistration& reg) {
      return sparse_anchor_config_to_dict(reg.getSparseAnchorConfig());
    })
    .def("get_observability_diagnostics", [] (LsqRegistration& reg) {
      const auto& diagnostics = reg.getObservabilityDiagnostics();
      py::dict info;
      info["raw_hessian"] = diagnostics.raw_hessian;
      info["regularized_hessian"] = diagnostics.regularized_hessian;
      info["eigenvalues"] = diagnostics.eigenvalues;
      info["ambiguity_scores"] = diagnostics.ambiguity_scores;
      info["auto_suppressed_mask"] = dof_mask_to_numpy(diagnostics.auto_suppressed_mask);
      info["hard_lock_mask"] = dof_mask_to_numpy(diagnostics.hard_lock_mask);
      info["preferred_ambiguous_mask"] = dof_mask_to_numpy(diagnostics.preferred_ambiguous_mask);
      info["condition_number"] = diagnostics.condition_number;
      info["estimated_rank"] = diagnostics.estimated_rank;
      info["geometry_eigenvalues"] = diagnostics.geometry_eigenvalues;
      info["smooth_regularization_weights"] = diagnostics.smooth_regularization_weights;
      info["geometry_estimated_rank"] = diagnostics.geometry_estimated_rank;
      info["geometry_condition_number"] = diagnostics.geometry_condition_number;
      return info;
    })
    .def("get_alignment_quality_report", [] (LsqRegistration& reg) {
      return alignment_quality_report_to_dict(reg.getAlignmentQualityReport());
    })
    .def("get_fitness_score", [] (LsqRegistration& reg, const double max_range) { return reg.getFitnessScore(max_range); })
    .def("align",
      [] (LsqRegistration& reg, const Eigen::Matrix4f& initial_guess) { 
        pcl::PointCloud<pcl::PointXYZ> aligned;
        reg.align(aligned, initial_guess);
        return reg.getFinalTransformation();
      }, py::arg("initial_guess") = Eigen::Matrix4f::Identity()
    )
  ;
  py::class_<FastGICP, LsqRegistration, std::shared_ptr<FastGICP>>(m, "FastGICP")
    .def(py::init())
    .def(py::pickle(
        [](const FastGICP &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getSourceRotationsq());
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            // FastGICP p(t[0].cast<std::string>());
            FastGICP p;

            /* Assign any additional state */
            // p.setExtra(t[1].cast<int>());

            return p;
        }
    ))
    .def("set_num_threads", &FastGICP::setNumThreads)
    .def("get_num_threads", &FastGICP::getNumThreads)
    .def("set_max_iterations", [] (FastGICP& gicp, int n) {
      gicp.setMaximumIterations(n);
    })
    .def("get_max_iterations", [] (FastGICP& gicp) {
      return gicp.getMaximumIterations();
    })
    .def("set_correspondence_randomness", &FastGICP::setCorrespondenceRandomness)
    .def("set_max_correspondence_distance", &FastGICP::setMaxCorrespondenceDistance)
    .def("set_max_knn_distance", &FastGICP::setKNNMaxDistance)
    .def("set_color_matching", [] (FastGICP& gicp, bool enabled, int candidate_count, double color_weight, double color_sigma) {
      gicp.setColorMatchingConfig(color_matching_config_from_args(enabled, candidate_count, color_weight, color_sigma));
    }, py::arg("enabled") = true, py::arg("candidate_count") = 5, py::arg("color_weight") = 0.0, py::arg("color_sigma") = 32.0)
    .def("set_color_matching_enabled", &FastGICP::setColorMatchingEnabled)
    .def("get_color_matching_config", [] (FastGICP& gicp) {
      const auto& config = gicp.getColorMatchingConfig();
      py::dict info;
      info["enable_color_matching"] = config.enable_color_matching;
      info["geometric_candidate_count"] = config.geometric_candidate_count;
      info["color_weight"] = config.color_weight;
      info["color_sigma"] = config.color_sigma;
      return info;
    })
    .def("set_source_colors", [] (FastGICP& gicp, const py::array& colors) {
      gicp.setSourceColors(numpy_to_vec3_list(colors));
    })
    .def("set_target_colors", [] (FastGICP& gicp, const py::array& colors) {
      gicp.setTargetColors(numpy_to_vec3_list(colors));
    })
    .def("clear_source_colors", &FastGICP::clearSourceColors)
    .def("clear_target_colors", &FastGICP::clearTargetColors)
    .def("get_source_rotationsq", [] (FastGICP& gicp){
      return py::array(gicp.getSourceRotationsqSize(), gicp.getSourceRotationsq().data());
      })
    .def("get_target_rotationsq", [] (FastGICP& gicp){
      return py::array(gicp.getTargetRotationsqSize(), gicp.getTargetRotationsq().data());
      })
    .def("get_source_scales", [] (FastGICP& gicp){
      return py::array(gicp.getSourceScaleSize(), gicp.getSourceScales().data());
      })
    .def("get_target_scales", [] (FastGICP& gicp){
      return py::array(gicp.getTargetScaleSize(), gicp.getTargetScales().data());
      })
    .def("calculate_source_covariance", &FastGICP::calculateSourceCovariance)
    .def("calculate_target_covariance", &FastGICP::calculateTargetCovariance)
    .def("calculate_target_covariance_withz", &FastGICP::calculateTargetCovarianceWithZ)
    .def("calculate_target_covariance_with_filter", &FastGICP::calculateTargetCovarianceWithFilter)
    .def("get_source_correspondence", [] (FastGICP& gicp){
    	return py::make_tuple(py::array(gicp.getSourceSize(), gicp.getSourceCorrespondences().data()), 
    				py::array(gicp.getSourceSize(), gicp.getSourceSqDistances().data()));
    })
    .def("set_source_covariances_fromqs", [] (FastGICP& gicp, py::array rotationsq, py::array scales){
    	if(py::len(rotationsq)/4!=py::len(scales)/3){ std::cerr<<"[set_source_covariances_fromqs] qs size not matched" <<std::endl; return;}
    	const auto input_rotationsq = rotationsq.cast<std::vector<float>>();
    	const auto input_scales = scales.cast<std::vector<float>>();
    	gicp.setSourceCovariances(input_rotationsq, input_scales);
    })
    .def("set_target_covariances_fromqs", [] (FastGICP& gicp, py::array rotationsq, py::array scales){
    	if(py::len(rotationsq)/4!=py::len(scales)/3){ std::cerr<<"[set_target_covariances_fromqs] qs size not matched" <<std::endl; return;}
    	const auto input_rotationsq = rotationsq.cast<std::vector<float>>();
    	const auto input_scales = scales.cast<std::vector<float>>();
    	gicp.setTargetCovariances(input_rotationsq, input_scales);
    })
    .def("set_source_z_values", [] (FastGICP& gicp, py::array z_values){
      const auto input_z_values = z_values.cast<std::vector<float>>();
    	gicp.setSourceZvalues(input_z_values);
    })
    .def("set_target_z_values", [] (FastGICP& gicp, py::array z_values){
      const auto input_z_values = z_values.cast<std::vector<float>>();
    	gicp.setTargetZvalues(input_z_values);
    })
    .def("set_source_filter", [] (FastGICP& gicp, py::int_ num_trackable, py::array filter){
      const auto input_filter = filter.cast<std::vector<int>>();
    	gicp.setSourceFilter(num_trackable, input_filter);
    })
    .def("set_target_filter", [] (FastGICP& gicp, py::int_ num_trackable, py::array filter){
      const auto input_filter = filter.cast<std::vector<int>>();
    	gicp.setTargetFilter(num_trackable, input_filter);
    })
  ;

  py::class_<FastVGICP, FastGICP, std::shared_ptr<FastVGICP>>(m, "FastVGICP")
    .def(py::init())
    .def("set_resolution", &FastVGICP::setResolution)
    .def("set_neighbor_search_method", [](FastVGICP& vgicp, const std::string& method) { vgicp.setNeighborSearchMethod(search_method(method)); })
    .def("get_voxel_mean_cov", [](FastVGICP& vgicp){ 
      using meanvec = typename std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>;
      using covvec = typename std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>;
      std::pair<meanvec, covvec> data = vgicp.getVoxelMeanCov();
//      return py::make_tuple(data.first, data.second);      
      py::list data0; py::list data1;
      for (const auto& d:data.first) data0.append(d);
      for (const auto& d:data.second) data1.append(d);
      return py::make_tuple(data0, data1);
    })
  ;

#ifdef USE_VGICP_CUDA
  py::class_<FastVGICPCuda, LsqRegistration, std::shared_ptr<FastVGICPCuda>>(m, "FastVGICPCuda")
    .def(py::init())
    .def("set_resolution", &FastVGICPCuda::setResolution)
    .def("set_neighbor_search_method",
      [](FastVGICPCuda& vgicp, const std::string& method, double radius) { vgicp.setNeighborSearchMethod(search_method(method), radius); }
      , py::arg("method") = "DIRECT1", py::arg("radius") = 1.5
    )
    .def("set_correspondence_randomness", &FastVGICPCuda::setCorrespondenceRandomness)
  ;

  py::class_<NDTCuda, LsqRegistration, std::shared_ptr<NDTCuda>>(m, "NDTCuda")
    .def(py::init())
    .def("set_neighbor_search_method",
      [](NDTCuda& ndt, const std::string& method, double radius) { ndt.setNeighborSearchMethod(search_method(method), radius); }
      , py::arg("method") = "DIRECT1", py::arg("radius") = 1.5
    )
    .def("set_resolution", &NDTCuda::setResolution)
  ;
#endif

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
