#include <vector>
#include <sstream>
#include <iostream>
#include <array>
#include <cmath>
#include <gtest/gtest.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

struct GICPTestBase : public testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using PointCloudConstPtr = pcl::PointCloud<pcl::PointXYZ>::ConstPtr;

  GICPTestBase() {}

  virtual void SetUp() {
    if (!load(data_directory)) {
      exit(1);
    }
  }

  bool load(const std::string& data_directory) {
    relative_pose.setIdentity();

    std::ifstream ifs(data_directory + "/relative.txt");
    if (!ifs) {
      return false;
    }

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        ifs >> relative_pose(i, j);
      }
    }

    auto target = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto source = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::io::loadPCDFile(data_directory + "/251370668.pcd", *target);
    pcl::io::loadPCDFile(data_directory + "/251371071.pcd", *source);
    if (target->empty() || source->empty()) {
      return true;
    }

    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(0.2, 0.2, 0.2);

    auto filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    voxelgrid.setInputCloud(target);
    voxelgrid.filter(*filtered);
    filtered.swap(target);

    voxelgrid.setInputCloud(source);
    voxelgrid.filter(*filtered);
    filtered.swap(source);

    this->target = target;
    this->source = source;

    return true;
  }

  Eigen::Vector2f pose_error(const Eigen::Matrix4f estimated) const {
    Eigen::Matrix4f delta = relative_pose.inverse() * estimated;
    double t_error = delta.block<3, 1>(0, 3).norm();
    double r_error = Eigen::AngleAxisf(delta.block<3, 3>(0, 0)).angle();
    return Eigen::Vector2f(t_error, r_error);
  }

  static std::string data_directory;

  PointCloudConstPtr target;
  PointCloudConstPtr source;
  Eigen::Matrix4f relative_pose;
};

std::string GICPTestBase::data_directory;

namespace {

pcl::PointCloud<pcl::PointXYZ>::Ptr make_vertical_plane(int y_count = 16, int z_count = 16, double step = 0.1) {
  auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
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

pcl::PointCloud<pcl::PointXYZ>::Ptr make_box_cloud(int count_per_axis = 5, double step = 0.12) {
  auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  cloud->reserve(count_per_axis * count_per_axis * count_per_axis);
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

pcl::PointCloud<pcl::PointXYZ>::Ptr transform_cloud(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
  const Eigen::Matrix4f& transform) {
  auto transformed = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::transformPointCloud(*cloud, *transformed, transform);
  return transformed;
}

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> identity_covariances(int count) {
  return std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>(count, Eigen::Matrix4d::Identity());
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> solid_colors(int count, const Eigen::Vector3d& color) {
  return std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(count, color);
}

}  // namespace

TEST_F(GICPTestBase, LoadCheck) {
  EXPECT_NE(target, nullptr);
  EXPECT_NE(source, nullptr);
  EXPECT_FALSE(target->empty());
  EXPECT_FALSE(source->empty());
}

using Parameters = std::tuple<const char*, bool>;
class AlignmentTest : public GICPTestBase, public testing::WithParamInterface<Parameters> {
public:
  pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr create_reg() {
    std::string method = std::get<0>(GetParam());
    int num_threads = std::get<1>(GetParam()) ? 4 : 1;

    if (method == "GICP") {
      auto gicp = pcl::make_shared<fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>>();
      gicp->setNumThreads(num_threads);
      gicp->swapSourceAndTarget();
      return gicp;
    } else if (method == "VGICP") {
      auto vgicp = pcl::make_shared<fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>>();
      vgicp->setNumThreads(num_threads);
      return vgicp;
    } else if (method == "VGICP_CUDA") {
#ifdef USE_VGICP_CUDA
      auto vgicp = pcl::make_shared<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>>();
      return vgicp;
#endif
      return nullptr;
    } else if (method == "NDT_CUDA") {
#ifdef USE_VGICP_CUDA
      auto ndt = pcl::make_shared<fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>>();
      return ndt;
#endif
      return nullptr;
    }

    std::cerr << "unknown registration method:" << method << std::endl;
    return nullptr;
  }

  void swap_source_and_target(pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr reg) {
    fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>* lsq_reg = dynamic_cast<fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>*>(reg.get());
    if (lsq_reg != nullptr) {
      lsq_reg->swapSourceAndTarget();
      return;
    }

    std::cerr << "failed to swap source and target" << std::endl;
  }
};

INSTANTIATE_TEST_SUITE_P(AlignmentTest2, AlignmentTest, testing::Combine(testing::Values("GICP", "VGICP", "VGICP_CUDA", "NDT_CUDA"), testing::Bool()), [](const auto& info) {
  std::stringstream sst;
  sst << std::get<0>(info.param) << (std::get<1>(info.param) ? "_MT" : "_ST");
  return sst.str();
});

TEST_P(AlignmentTest, test) {
  const double t_tol = 0.05;
  const double r_tol = 1.0 * M_PI / 180.0;

  pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr reg = create_reg();
  if (reg == nullptr) {
    std::cout << "[          ] SKIP TEST" << std::endl;
    return;
  }

  // forward test
  auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  reg->setInputTarget(target);
  reg->setInputSource(source);
  reg->align(*aligned);

  Eigen::Vector2f errors = pose_error(reg->getFinalTransformation());
  EXPECT_LT(errors[0], t_tol) << "FORWARD TEST";
  EXPECT_LT(errors[1], r_tol) << "FORWARD TEST";
  EXPECT_TRUE(reg->hasConverged()) << "FORWARD TEST";

  // backward test
  reg->setInputTarget(source);
  reg->setInputSource(target);
  reg->align(*aligned);

  errors = pose_error(reg->getFinalTransformation().inverse());
  EXPECT_LT(errors[0], t_tol) << "BACKWARD TEST";
  EXPECT_LT(errors[1], r_tol) << "BACKWARD TEST";
  EXPECT_TRUE(reg->hasConverged()) << "BACKWARD TEST";

  // swap and set source
  reg = create_reg();
  reg->setInputSource(target);
  swap_source_and_target(reg);
  reg->setInputSource(source);
  reg->align(*aligned);

  errors = pose_error(reg->getFinalTransformation());
  EXPECT_LT(errors[0], t_tol) << "SWAP AND SET SOURCE TEST";
  EXPECT_LT(errors[1], r_tol) << "SWAP AND SET SOURCE TEST";
  EXPECT_TRUE(reg->hasConverged()) << "SWAP AND SET SOURCE TEST";

  // swap and set target
  reg = create_reg();
  reg->setInputTarget(source);
  swap_source_and_target(reg);  // source:target, target:source
  reg->setInputTarget(target);
  reg->align(*aligned);

  errors = pose_error(reg->getFinalTransformation());
  EXPECT_LT(errors[0], t_tol) << "SWAP AND SET TARGET TEST";
  EXPECT_LT(errors[1], r_tol) << "SWAP AND SET TARGET TEST";
  EXPECT_TRUE(reg->hasConverged()) << "SWAP AND SET TARGET TEST";
}

TEST(RobustGICPTest, ObservabilityDiagnosticsDetectPlanarAmbiguity) {
  auto target = make_vertical_plane();
  auto source = make_vertical_plane();

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.setMaxCorrespondenceDistance(1.0);
  reg.setObservabilityCheck(true);
  reg.setEnableObservabilityDiagnostics(true);

  reg.evaluateCost(Eigen::Matrix4f::Identity());
  const auto& diagnostics = reg.getObservabilityDiagnostics();

  EXPECT_LT(diagnostics.estimated_rank, fast_gicp::kLsqDof);
  EXPECT_GT(diagnostics.ambiguity_scores[5], 0.1);
  EXPECT_TRUE(std::isinf(diagnostics.condition_number) || diagnostics.condition_number > 1e3);
}

TEST(RobustGICPTest, HardLockKeepsTranslationZFixed) {
  auto target = make_box_cloud();

  Eigen::Matrix4f true_transform = Eigen::Matrix4f::Identity();
  true_transform(0, 3) = 0.2f;
  true_transform(1, 3) = -0.1f;
  true_transform(2, 3) = 0.3f;
  auto source = transform_cloud(target, true_transform.inverse());

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.setMaxCorrespondenceDistance(1.0);
  reg.setHardLockMask({{0, 0, 0, 0, 0, 1}});

  pcl::PointCloud<pcl::PointXYZ> aligned;
  reg.align(aligned);

  const Eigen::Matrix4f estimated = reg.getFinalTransformation();
  EXPECT_NEAR(estimated(2, 3), 0.0f, 1e-5f);
  EXPECT_NEAR(estimated(0, 3), true_transform(0, 3), 0.05f);
}

TEST(RobustGICPTest, SparseAnchorsBiasAmbiguousTranslation) {
  auto target = make_vertical_plane();
  auto source = make_vertical_plane();

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> no_anchor_reg;
  no_anchor_reg.setInputTarget(target);
  no_anchor_reg.setInputSource(source);
  no_anchor_reg.setMaxCorrespondenceDistance(1.0);

  pcl::PointCloud<pcl::PointXYZ> aligned;
  no_anchor_reg.align(aligned);
  const double no_anchor_z = no_anchor_reg.getFinalTransformation()(2, 3);

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> anchor_reg;
  anchor_reg.setInputTarget(target);
  anchor_reg.setInputSource(source);
  anchor_reg.setMaxCorrespondenceDistance(1.0);

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> source_anchors;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> target_anchors;
  std::vector<double> weights;
  for (int i = 0; i < 4; i++) {
    const auto& point = source->at(i * 10);
    source_anchors.emplace_back(point.x, point.y, point.z);
    target_anchors.emplace_back(point.x, point.y, point.z + 0.2);
    weights.push_back(50.0);
  }

  anchor_reg.setSparseAnchorCorrespondences(source_anchors, target_anchors, weights);
  anchor_reg.align(aligned);
  const double anchor_z = anchor_reg.getFinalTransformation()(2, 3);

  EXPECT_LT(std::abs(no_anchor_z), 0.05);
  EXPECT_GT(anchor_z, 0.1);
}

TEST(RobustGICPTest, ColorMatchingChangesCorrespondenceSelection) {
  auto target = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  target->push_back(pcl::PointXYZ(0.05f, 0.0f, 0.0f));
  target->push_back(pcl::PointXYZ(0.10f, 0.0f, 0.0f));

  auto source = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
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
  EXPECT_EQ(reg.getSourceCorrespondences()[0], 1);
}

int main(int argc, char** argv) {
  GICPTestBase::data_directory = argv[1];
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
