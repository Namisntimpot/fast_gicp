#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <fast_gicp/gicp/fast_gicp.hpp>

namespace {

using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using CloudPtr = Cloud::Ptr;

struct Dataset {
  CloudPtr target;
  CloudPtr source;
  Eigen::Matrix4f relative_pose = Eigen::Matrix4f::Identity();
};

Dataset load_dataset(const std::string& data_directory) {
  Dataset dataset;
  dataset.target = pcl::make_shared<Cloud>();
  dataset.source = pcl::make_shared<Cloud>();

  std::ifstream ifs(data_directory + "/relative.txt");
  if (!ifs) {
    throw std::runtime_error("failed to open relative.txt");
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      ifs >> dataset.relative_pose(i, j);
    }
  }

  if (pcl::io::loadPCDFile(data_directory + "/251370668.pcd", *dataset.target) != 0 ||
      pcl::io::loadPCDFile(data_directory + "/251371071.pcd", *dataset.source) != 0) {
    throw std::runtime_error("failed to load benchmark point clouds");
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
  const double t_error = delta.block<3, 1>(0, 3).norm();
  const double r_error = Eigen::AngleAxisf(delta.block<3, 3>(0, 0)).angle();
  return Eigen::Vector2f(t_error, r_error);
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> make_anchor_points(
  const CloudPtr& source,
  const Eigen::Matrix4f& relative_pose,
  int stride) {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> anchors;
  anchors.reserve(source->size() / stride + 1);
  for (int i = 0; i < source->size(); i += stride) {
    Eigen::Vector4f point = source->at(i).getVector4fMap();
    Eigen::Vector4f transformed = relative_pose * point;
    anchors.emplace_back(transformed.x(), transformed.y(), transformed.z());
  }
  return anchors;
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> source_anchor_points(const CloudPtr& source, int stride) {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> anchors;
  anchors.reserve(source->size() / stride + 1);
  for (int i = 0; i < source->size(); i += stride) {
    const auto& point = source->at(i);
    anchors.emplace_back(point.x, point.y, point.z);
  }
  return anchors;
}

struct BenchmarkResult {
  std::string name;
  double mean_ms = 0.0;
  double fitness_score = 0.0;
};

BenchmarkResult run_case(
  const std::string& name,
  const Dataset& dataset,
  int iterations,
  const std::function<void(fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>&)>& configure) {
  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
  reg.swapSourceAndTarget();
  reg.setInputTarget(dataset.target);
  reg.setInputSource(dataset.source);
  reg.setMaxCorrespondenceDistance(2.0);
  reg.setNumThreads(4);
  configure(reg);

  pcl::PointCloud<pcl::PointXYZ> aligned;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    reg.align(aligned);
  }
  auto end = std::chrono::steady_clock::now();

  BenchmarkResult result;
  result.name = name;
  result.mean_ms =
    std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count() /
    static_cast<double>(iterations);
  result.fitness_score = reg.getFitnessScore();
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string data_directory = argc > 1 ? argv[1] : "data";
  const int iterations = argc > 2 ? std::max(1, std::atoi(argv[2])) : 20;

  const Dataset dataset = load_dataset(data_directory);
  const auto source_anchors = source_anchor_points(dataset.source, 8);
  const auto target_anchors = make_anchor_points(dataset.source, dataset.relative_pose, 8);
  const std::vector<double> anchor_weights(source_anchors.size(), 10.0);

  std::vector<BenchmarkResult> results;
  results.push_back(run_case("baseline_default", dataset, iterations, [](auto&) {}));
  results.push_back(run_case("robust_observability", dataset, iterations, [](auto& reg) {
    reg.setObservabilityCheck(true);
    reg.setAutoSoftPriorStrength(0.02);
    reg.setPreferredAmbiguousMask({{0, 0, 0, 0, 0, 1}});
  }));
  results.push_back(run_case("robust_observability_anchor", dataset, iterations, [&](auto& reg) {
    reg.setObservabilityCheck(true);
    reg.setAutoSoftPriorStrength(0.02);
    reg.setPreferredAmbiguousMask({{0, 0, 0, 0, 0, 1}});
    reg.setSparseAnchorCorrespondences(source_anchors, target_anchors, anchor_weights);
  }));

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "name,mean_ms,fitness_score" << std::endl;
  for (const auto& result : results) {
    std::cout << result.name << ","
              << result.mean_ms << ","
              << result.fitness_score << std::endl;
  }

  return 0;
}
