#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/registration.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

namespace {

using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using CloudPtr = Cloud::Ptr;

struct TestFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void expect_true(bool value, const std::string& message) {
  if (!value) {
    throw TestFailure(message);
  }
}

void expect_near(double lhs, double rhs, double tol, const std::string& message) {
  if (std::abs(lhs - rhs) > tol) {
    throw TestFailure(message + " lhs=" + std::to_string(lhs) + " rhs=" + std::to_string(rhs));
  }
}

struct Dataset {
  CloudPtr target = pcl::make_shared<Cloud>();
  CloudPtr source = pcl::make_shared<Cloud>();
  Eigen::Matrix4f relative_pose = Eigen::Matrix4f::Identity();
};

Dataset load_dataset(const std::string& data_directory) {
  Dataset dataset;

  std::ifstream ifs(data_directory + "/relative.txt");
  if (!ifs) {
    throw TestFailure("failed to open relative.txt");
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      ifs >> dataset.relative_pose(i, j);
    }
  }

  if (pcl::io::loadPCDFile(data_directory + "/251370668.pcd", *dataset.target) != 0 ||
      pcl::io::loadPCDFile(data_directory + "/251371071.pcd", *dataset.source) != 0) {
    throw TestFailure("failed to load benchmark clouds");
  }

  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.2f, 0.2f, 0.2f);
  auto filtered = pcl::make_shared<Cloud>();
  voxelgrid.setInputCloud(dataset.target);
  voxelgrid.filter(*filtered);
  dataset.target.swap(filtered);

  filtered = pcl::make_shared<Cloud>();
  voxelgrid.setInputCloud(dataset.source);
  voxelgrid.filter(*filtered);
  dataset.source.swap(filtered);

  return dataset;
}

Eigen::Vector2f pose_error(const Eigen::Matrix4f& reference, const Eigen::Matrix4f& estimated) {
  Eigen::Matrix4f delta = reference.inverse() * estimated;
  return Eigen::Vector2f(
    delta.block<3, 1>(0, 3).norm(),
    Eigen::AngleAxisf(delta.block<3, 3>(0, 0)).angle());
}

CloudPtr make_vertical_plane(int y_count = 16, int z_count = 16, double step = 0.1) {
  auto cloud = pcl::make_shared<Cloud>();
  cloud->reserve(y_count * z_count);
  for (int iy = 0; iy < y_count; iy++) {
    for (int iz = 0; iz < z_count; iz++) {
      pcl::PointXYZ point;
      point.x = 0.0f;
      point.y = static_cast<float>((iy - y_count / 2) * step);
      point.z = static_cast<float>((iz - z_count / 2) * step);
      cloud->push_back(point);
    }
  }
  return cloud;
}

CloudPtr make_box_cloud(int count_per_axis = 5, double step = 0.12) {
  auto cloud = pcl::make_shared<Cloud>();
  for (int ix = 0; ix < count_per_axis; ix++) {
    for (int iy = 0; iy < count_per_axis; iy++) {
      for (int iz = 0; iz < count_per_axis; iz++) {
        pcl::PointXYZ point;
        point.x = static_cast<float>((ix - count_per_axis / 2) * step);
        point.y = static_cast<float>((iy - count_per_axis / 2) * step);
        point.z = static_cast<float>((iz - count_per_axis / 2) * step);
        cloud->push_back(point);
      }
    }
  }
  return cloud;
}

CloudPtr transform_cloud(const CloudPtr& cloud, const Eigen::Matrix4f& transform) {
  auto transformed = pcl::make_shared<Cloud>();
  pcl::transformPointCloud(*cloud, *transformed, transform);
  return transformed;
}

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> identity_covariances(int count) {
  return std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>(count, Eigen::Matrix4d::Identity());
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> solid_colors(int count, const Eigen::Vector3d& color) {
  return std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(count, color);
}

bool has_reason(const fast_gicp::AlignmentQualityReport& report, const std::string& reason) {
  return std::find(report.rejection_reasons.begin(), report.rejection_reasons.end(), reason) != report.rejection_reasons.end();
}

void run_alignment_regression() {
  auto target = make_box_cloud();
  Eigen::Matrix4f reference = Eigen::Matrix4f::Identity();
  reference(0, 3) = 0.2f;
  reference(1, 3) = -0.1f;
  reference(2, 3) = 0.15f;
  auto source = transform_cloud(target, reference.inverse());

  {
    std::cout << "  [CASE] FastGICP baseline" << std::endl;
    fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
    reg.setInputTarget(target);
    reg.setInputSource(source);
    reg.setMaxCorrespondenceDistance(1.0);

    Cloud aligned;
    reg.align(aligned);

    expect_true(reg.getFinalTransformation().array().isFinite().all(), "FastGICP baseline produced invalid transform");
  }

  {
    std::cout << "  [CASE] FastVGICP baseline" << std::endl;
    fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
    reg.setInputTarget(target);
    reg.setInputSource(source);
    reg.setMaxCorrespondenceDistance(1.0);

    Cloud aligned;
    reg.align(aligned);

    expect_true(reg.getFinalTransformation().array().isFinite().all(), "FastVGICP baseline produced invalid transform");
  }
}

void run_observability_checks() {
  auto target = pcl::make_shared<Cloud>();
  target->push_back(pcl::PointXYZ(0.0f, 0.0f, 1.0f));
  auto source = pcl::make_shared<Cloud>();
  source->push_back(pcl::PointXYZ(0.0f, 0.0f, 1.0f));

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.setMaxCorrespondenceDistance(1.0);
  reg.setObservabilityCheck(true);
  reg.setEnableObservabilityDiagnostics(true);
  reg.setObservabilityEigenThresholds(0.1, 1e-6);

  Eigen::Matrix4d plane_cov = Eigen::Matrix4d::Zero();
  plane_cov(0, 0) = 1e-3;
  plane_cov(1, 1) = 1.0;
  plane_cov(2, 2) = 1.0;
  plane_cov(3, 3) = 1.0;
  reg.setTargetCovariances({plane_cov});
  reg.setSourceCovariances({plane_cov});
  reg.evaluateCost(Eigen::Matrix4f::Identity());

  const auto& diagnostics = reg.getObservabilityDiagnostics();
  std::cout << "    rank=" << diagnostics.estimated_rank
            << " eig=" << diagnostics.eigenvalues.transpose()
            << " ambiguity=" << diagnostics.ambiguity_scores.transpose() << std::endl;
  expect_true(diagnostics.estimated_rank < fast_gicp::kLsqDof, "planar scene should be rank deficient");
  expect_true(diagnostics.ambiguity_scores[5] > 0.1, "trans_z should be ambiguous for planar covariance");
}

void run_hard_lock_check() {
  auto target = make_box_cloud();
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform(0, 3) = 0.2f;
  transform(1, 3) = -0.1f;
  transform(2, 3) = 0.3f;
  auto source = transform_cloud(target, transform.inverse());

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.setMaxCorrespondenceDistance(1.0);
  reg.setHardLockMask({{0, 0, 0, 0, 0, 1}});

  Cloud aligned;
  reg.align(aligned);
  const Eigen::Matrix4f estimated = reg.getFinalTransformation();
  expect_near(estimated(2, 3), 0.0, 1e-5, "hard-lock should keep z fixed");
  expect_near(estimated(0, 3), transform(0, 3), 0.12, "hard-lock should still recover x");
}

void run_sparse_anchor_check() {
  auto target = make_vertical_plane();
  auto source = make_vertical_plane();

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.setMaxCorrespondenceDistance(1.0);

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> source_anchors;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> target_anchors;
  std::vector<double> weights;
  for (int i = 0; i < 4; i++) {
    const auto& point = source->at(i * 10);
    source_anchors.emplace_back(point.x, point.y, point.z);
    target_anchors.emplace_back(point.x, point.y, point.z + 0.2);
    weights.push_back(50.0);
  }

  reg.setSparseAnchorCorrespondences(source_anchors, target_anchors, weights);

  Cloud aligned;
  reg.align(aligned);
  expect_true(reg.getFinalTransformation()(2, 3) > 0.1, "sparse anchors should bias z translation");
}

void run_color_matching_check() {
  auto target = pcl::make_shared<Cloud>();
  target->push_back(pcl::PointXYZ(0.05f, 0.0f, 0.0f));
  target->push_back(pcl::PointXYZ(0.10f, 0.0f, 0.0f));
  auto source = pcl::make_shared<Cloud>();
  source->push_back(pcl::PointXYZ(0.0f, 0.0f, 0.0f));

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.setMaxCorrespondenceDistance(1.0);
  reg.setSourceCovariances(identity_covariances(1));
  reg.setTargetCovariances(identity_covariances(2));
  reg.setSourceColors(solid_colors(1, Eigen::Vector3d(255.0, 0.0, 0.0)));
  reg.setTargetColors({
    Eigen::Vector3d(0.0, 0.0, 255.0),
    Eigen::Vector3d(255.0, 0.0, 0.0),
  });
  reg.setColorMatchingConfig({true, 2, 1.0, 16.0});
  reg.evaluateCost(Eigen::Matrix4f::Identity());

  expect_true(reg.getSourceCorrespondences()[0] == 1, "color matching should prefer the color-consistent target");
}

void run_alignment_quality_report_check() {
  auto target = make_box_cloud();
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform(0, 3) = 0.18f;
  transform(1, 3) = -0.05f;
  transform(2, 3) = 0.09f;
  auto source = transform_cloud(target, transform.inverse());

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.setMaxCorrespondenceDistance(1.0);

  fast_gicp::AlignmentQualityConfig quality_config;
  quality_config.enable_suggested_gating = true;
  quality_config.min_matched_count = 10;
  quality_config.min_rank = 3;
  reg.setAlignmentQualityConfig(quality_config);

  Cloud aligned;
  reg.align(aligned);

  const auto& report = reg.getAlignmentQualityReport();
  expect_true(report.valid, "alignment quality report should be valid after align");
  expect_true(report.has_match_statistics, "FastGICP should populate match statistics");
  expect_true(report.matched_count > 0, "alignment quality report should have matches");
  expect_true(report.correspondence_count == report.matched_count, "FastGICP correspondence count should match matched count");
  expect_true(std::isfinite(report.mean_sq_distance), "mean_sq_distance should be finite");
  expect_true(report.matched_ratio > 0.0 && report.matched_ratio <= 1.0, "matched_ratio should be in (0, 1]");
  expect_true(report.gating_evaluated, "gating should be evaluated when enabled");
}

void run_alignment_quality_gating_check() {
  auto target = make_vertical_plane();
  auto source = make_vertical_plane();

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.setMaxCorrespondenceDistance(1.0);

  fast_gicp::AlignmentQualityConfig quality_config;
  quality_config.enable_suggested_gating = true;
  quality_config.max_dof_ambiguity = {std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity(),
                                      0.2};
  reg.setAlignmentQualityConfig(quality_config);

  Cloud aligned;
  reg.align(aligned);

  const auto& report = reg.getAlignmentQualityReport();
  expect_true(!report.suggested_accept, "planar ambiguity should be rejected by suggested gating");
  expect_true(has_reason(report, "trans_z_ambiguous"), "planar ambiguity should flag trans_z");
}

void run_alignment_quality_anchor_and_vgicp_check() {
  auto target = make_vertical_plane();
  auto source = make_vertical_plane();

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
  gicp.setInputTarget(target);
  gicp.setInputSource(source);
  gicp.setMaxCorrespondenceDistance(1.0);

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> source_anchors;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> target_anchors;
  std::vector<double> weights;
  for (int i = 0; i < 4; i++) {
    const auto& point = source->at(i * 10);
    source_anchors.emplace_back(point.x, point.y, point.z);
    target_anchors.emplace_back(point.x, point.y, point.z + 0.2);
    weights.push_back(50.0);
  }
  gicp.setSparseAnchorCorrespondences(source_anchors, target_anchors, weights);

  Cloud aligned;
  gicp.align(aligned);

  const auto& anchor_report = gicp.getAlignmentQualityReport();
  expect_true(anchor_report.used_sparse_anchors, "anchor report should flag sparse anchors");
  expect_true(anchor_report.has_anchor_statistics, "anchor statistics should be populated");
  expect_true(anchor_report.anchor_count == 4, "anchor count should match configured anchors");
  expect_true(std::isfinite(anchor_report.anchor_mean_residual), "anchor mean residual should be finite");

  auto box_target = make_box_cloud();
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform(0, 3) = 0.15f;
  transform(1, 3) = -0.08f;
  transform(2, 3) = 0.06f;
  auto box_source = transform_cloud(box_target, transform.inverse());

  fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> vgicp;
  vgicp.setInputTarget(box_target);
  vgicp.setInputSource(box_source);
  vgicp.setMaxCorrespondenceDistance(1.0);

  vgicp.align(aligned);
  const auto& vgicp_report = vgicp.getAlignmentQualityReport();
  expect_true(vgicp_report.valid, "FastVGICP should produce an alignment quality report");
  expect_true(vgicp_report.has_match_statistics, "FastVGICP should populate match statistics");
  expect_true(vgicp_report.correspondence_count >= vgicp_report.matched_count, "voxel correspondences should dominate matched points");
  expect_true(vgicp_report.matched_count > 0, "FastVGICP should match at least one point");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::cout << "[RUN] alignment regression" << std::endl;
    run_alignment_regression();
    std::cout << "[RUN] observability" << std::endl;
    run_observability_checks();
    std::cout << "[RUN] hard lock" << std::endl;
    run_hard_lock_check();
    std::cout << "[RUN] sparse anchors" << std::endl;
    run_sparse_anchor_check();
    std::cout << "[RUN] color matching" << std::endl;
    run_color_matching_check();
    std::cout << "[RUN] alignment quality report" << std::endl;
    run_alignment_quality_report_check();
    std::cout << "[RUN] alignment quality gating" << std::endl;
    run_alignment_quality_gating_check();
    std::cout << "[RUN] alignment quality anchor/vgicp" << std::endl;
    run_alignment_quality_anchor_and_vgicp_check();
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << std::endl;
    return 1;
  }

  std::cout << "[PASS] fast_gicp self-checks completed" << std::endl;
  return 0;
}
