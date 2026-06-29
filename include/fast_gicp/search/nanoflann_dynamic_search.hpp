// Append-only, tombstone-supporting target KD-tree for FastGICP.
//
// Wraps nanoflann::KDTreeSingleIndexDynamicAdaptor with float xyz storage
// and a parallel alive[] mask. Indices into our internal point list are
// stable across the lifetime of one "epoch" (until rebuildFromScratch()).
//
// Used by FastGICP when target_kdtree_mode_ == "incremental". Knn returns
// the same nearest-target index as a static PCL KdTreeFLANN rebuilt on the
// live point set, modulo tie-break ordering.

#ifndef FAST_GICP_NANOFLANN_DYNAMIC_SEARCH_HPP
#define FAST_GICP_NANOFLANN_DYNAMIC_SEARCH_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <nanoflann.hpp>

namespace fast_gicp {

template <typename PointT>
class NanoflannDynamicSearch {
 public:
  using PointCloud = pcl::PointCloud<PointT>;
  using PointCloudConstPtr = typename PointCloud::ConstPtr;

  struct DatasetAdaptor {
    const NanoflannDynamicSearch* parent = nullptr;
    inline std::size_t kdtree_get_point_count() const { return parent->points_xyz_.size() / 3; }
    inline float kdtree_get_pt(std::size_t i, std::size_t d) const { return parent->points_xyz_[3 * i + d]; }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
  };

  // Double-precision L2 metric: promotes the float coords to double BEFORE subtracting, so the NN
  // distance ordering is computed entirely in float64. This makes the K=1 nearest the TRUE
  // float64-nearest — identical to what the CANONICAL_TIEBREAK re-rank picks — WITHOUT the K>=2
  // search + per-candidate re-rank cost. float32 distance rounding at near-ties (equidistant target
  // points) is what made plain nanoflann pick wrong correspondences (room/outer-wall overlap on the
  // long zhiyuan_all); the double subtraction removes the ambiguity. Point storage stays float
  // (no memory cost); only the ~3 subtract/square ops per node visit run in double.
  struct L2_Double_Adaptor {
    using ElementType = float;
    using DistanceType = double;
    const DatasetAdaptor& data_source;
    explicit L2_Double_Adaptor(const DatasetAdaptor& ds) : data_source(ds) {}
    inline DistanceType evalMetric(const float* a, const std::uint32_t b_idx, std::size_t size) const {
      DistanceType result = DistanceType();
      for (std::size_t i = 0; i < size; ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(data_source.kdtree_get_pt(b_idx, i));
        result += diff * diff;
      }
      return result;
    }
    template <typename U, typename V>
    inline DistanceType accum_dist(const U a, const V b, const std::size_t) const {
      const double diff = static_cast<double>(a) - static_cast<double>(b);
      return diff * diff;
    }
  };

  using DistanceT = L2_Double_Adaptor;
  using TreeT = nanoflann::KDTreeSingleIndexDynamicAdaptor<DistanceT, DatasetAdaptor, 3, std::uint32_t>;

  NanoflannDynamicSearch() : adaptor_{this} {}

  // Full rebuild from a PCL cloud. Discards all tombstones.
  void setInputCloud(const PointCloudConstPtr& cloud) {
    points_xyz_.clear();
    alive_.clear();
    if (!cloud || cloud->empty()) {
      rebuildTree();
      return;
    }
    const std::size_t n = cloud->size();
    points_xyz_.reserve(3 * n);
    alive_.assign(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
      points_xyz_.push_back(cloud->at(i).x);
      points_xyz_.push_back(cloud->at(i).y);
      points_xyz_.push_back(cloud->at(i).z);
    }
    rebuildTree();
  }

  // Append xyz (interleaved x0,y0,z0,x1,y1,z1,...). New indices = old size .. old size + n - 1.
  // Returns the first new index.
  std::size_t appendPoints(const float* xyz, std::size_t n) {
    if (n == 0) return points_xyz_.size() / 3;
    const std::size_t old_n = points_xyz_.size() / 3;
    points_xyz_.insert(points_xyz_.end(), xyz, xyz + 3 * n);
    alive_.insert(alive_.end(), n, static_cast<std::uint8_t>(1));
    if (!tree_) {
      rebuildTree();
    } else {
      tree_->addPoints(static_cast<std::uint32_t>(old_n), static_cast<std::uint32_t>(old_n + n - 1));
      live_count_ += n;
    }
    return old_n;
  }

  // Tombstone the given indices. Returns number of newly removed.
  std::size_t removePoints(const std::int32_t* indices, std::size_t n) {
    if (!tree_) return 0;
    const std::size_t total = points_xyz_.size() / 3;
    std::size_t removed = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const std::int32_t idx = indices[i];
      if (idx < 0 || static_cast<std::size_t>(idx) >= total) continue;
      if (!alive_[idx]) continue;
      alive_[idx] = 0;
      tree_->removePoint(static_cast<std::size_t>(idx));
      ++removed;
    }
    live_count_ = (live_count_ >= removed) ? live_count_ - removed : 0;
    return removed;
  }

  // Force a complete rebuild on current live points only; tombstones are
  // garbage-collected and indices are compacted to [0, live_count_).
  // Returns a mapping old_idx -> new_idx (or -1 if tombstoned). Caller is
  // responsible for re-indexing its parallel attribute arrays.
  std::vector<std::int32_t> compact() {
    const std::size_t old_total = points_xyz_.size() / 3;
    std::vector<std::int32_t> remap(old_total, -1);
    std::vector<float> new_xyz;
    new_xyz.reserve(3 * live_count_);
    std::vector<std::uint8_t> new_alive;
    new_alive.reserve(live_count_);
    std::size_t new_idx = 0;
    for (std::size_t i = 0; i < old_total; ++i) {
      if (!alive_[i]) continue;
      new_xyz.push_back(points_xyz_[3 * i + 0]);
      new_xyz.push_back(points_xyz_[3 * i + 1]);
      new_xyz.push_back(points_xyz_[3 * i + 2]);
      new_alive.push_back(1);
      remap[i] = static_cast<std::int32_t>(new_idx++);
    }
    points_xyz_.swap(new_xyz);
    alive_.swap(new_alive);
    rebuildTree();
    return remap;
  }

  // KNN. Returns number of neighbors actually found (<= k).
  int nearestKSearch(const PointT& point, int k, std::vector<int>& indices, std::vector<float>& sq_dists) const {
    indices.assign(k, -1);
    sq_dists.assign(k, std::numeric_limits<float>::max());
    if (!tree_ || live_count_ == 0) return 0;
    const float query[3] = {point.x, point.y, point.z};
    std::vector<std::uint32_t> out_idx(k);
    std::vector<double> out_dist(k);  // tree DistanceType is double (L2_Double_Adaptor)
    nanoflann::KNNResultSet<double, std::uint32_t> result(static_cast<std::size_t>(k));
    result.init(out_idx.data(), out_dist.data());
    tree_->findNeighbors(result, query);
    const int found = static_cast<int>(result.size());
    for (int i = 0; i < found; ++i) {
      indices[i] = static_cast<int>(out_idx[i]);
      sq_dists[i] = static_cast<float>(out_dist[i]);  // float64 dist -> float for the GICP gate/weight
    }
    return found;
  }

  std::size_t totalCount() const { return points_xyz_.size() / 3; }
  std::size_t liveCount() const { return live_count_; }
  std::size_t tombstonedCount() const { return totalCount() - live_count_; }
  bool isLive(std::size_t idx) const { return idx < alive_.size() && alive_[idx] != 0; }

  // For diagnostics and identity ties between Python/C++.
  const std::vector<float>& pointsXyzFlat() const { return points_xyz_; }
  const std::vector<std::uint8_t>& aliveMask() const { return alive_; }

 private:
  void rebuildTree() {
    const std::size_t n = points_xyz_.size() / 3;
    // The maximumPointCount governs how many internal sub-trees nanoflann
    // pre-allocates (one per bit, so log2(max) + 1 trees). Pick a value that
    // is comfortably larger than any expected live set; cost is just a few
    // empty placeholder trees.
    const std::size_t max_capacity = std::max<std::size_t>(1u << 22, n * 4u);
    nanoflann::KDTreeSingleIndexAdaptorParams params(20 /*leaf_max_size*/);
    tree_.reset(new TreeT(3, adaptor_, params, max_capacity));
    // Constructor auto-adds [0, n-1] from adaptor; but it counts dead points
    // too. Tombstone the dead ones again.
    live_count_ = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (alive_[i]) {
        ++live_count_;
      } else {
        tree_->removePoint(i);
      }
    }
  }

  std::vector<float> points_xyz_;  // 3*N interleaved
  std::vector<std::uint8_t> alive_;
  DatasetAdaptor adaptor_;
  std::unique_ptr<TreeT> tree_;
  std::size_t live_count_ = 0;
};

}  // namespace fast_gicp

#endif  // FAST_GICP_NANOFLANN_DYNAMIC_SEARCH_HPP
