cmake_minimum_required(VERSION 3.10.0)
project(fast_gicp)

option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" OFF)
option(BUILD_apps "Build application programs" ON)
option(BUILD_test "Build test programs" OFF)
option(BUILD_benchmarks "Build benchmark programs" OFF)
option(BUILD_PYTHON_BINDINGS "Build python bindings" OFF)

set(CMAKE_BUILD_TYPE "Release")

find_package(PCL REQUIRED COMPONENTS common io filters search registration)
find_package(Eigen3 REQUIRED)
add_definitions(${PCL_DEFINITIONS})

if(DEFINED ENV{ROS_VERSION})
  set(ROS_VERSION $ENV{ROS_VERSION})
endif()

if(NOT BUILD_PYTHON_BINDINGS)
  if(${ROS_VERSION})
    if(${ROS_VERSION} EQUAL 1)
      find_package(catkin)
    elseif (${ROS_VERSION} EQUAL 2)
      find_package(ament_cmake)
    endif()
  endif()
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(BUILD_VGICP_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  link_directories(${CUDA_LIBRARY_DIRS})
endif()

###################################
## catkin specific configuration ##
###################################
if(catkin_FOUND)
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES fast_gicp
  )
endif()

###########
## Build ##
###########

add_library(fast_gicp SHARED
  src/fast_gicp/gicp/lsq_registration.cpp
  src/fast_gicp/gicp/fast_gicp.cpp
  src/fast_gicp/gicp/fast_gicp_st.cpp
  src/fast_gicp/gicp/fast_vgicp.cpp
  src/fast_gicp/gicp/dynamic_rejection.cpp
)
target_link_libraries(fast_gicp
  ${PCL_LIBRARIES}
)
if (OPENMP_FOUND)
    if (TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(fast_gicp OpenMP::OpenMP_CXX)
    endif ()
endif ()
target_include_directories(fast_gicp PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

### APPS ###
if(BUILD_apps)
  add_executable(gicp_align src/align.cpp)
  add_dependencies(gicp_align fast_gicp)
  target_link_libraries(gicp_align
    ${PCL_LIBRARIES}
    fast_gicp
  )

  add_executable(gicp_kitti src/kitti.cpp)
  add_dependencies(gicp_kitti fast_gicp)
  target_link_libraries(gicp_kitti
    ${PCL_LIBRARIES}
    fast_gicp
  )
endif()

### Python bindings ###
if(BUILD_PYTHON_BINDINGS)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(pygicp
    src/python/main.cpp
  )
  target_include_directories(pygicp PUBLIC
    include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )
  target_link_libraries(pygicp PRIVATE
    fast_gicp
  )
endif()

### CUDA ###
option(USE_CUVS "Build the cuVS KNN backend for FastGICPCuda (requires cuVS)" OFF)
if(BUILD_VGICP_CUDA)
  # CUDA 12.x defaults to C++17 in nvcc; thrust 2.x requires C++17 to avoid
  # thrust::pair ambiguity with libcudacxx. Force explicitly so old downstreams
  # don't accidentally trigger C++14 paths in thrust headers.
  set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr;-std=c++17;--expt-extended-lambda")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
  # Suppress noisy Eigen __host__/__device__ defaulted-function warnings.
  list(APPEND CUDA_NVCC_FLAGS "-Xcudafe;--diag_suppress=20012")
  # Target Blackwell + Ada + Ampere by default. CUDA 12.8 supports sm_120.
  if(NOT DEFINED FAST_GICP_CUDA_ARCH)
    set(FAST_GICP_CUDA_ARCH "120" CACHE STRING "CUDA arch (e.g. 75;86;89;120)")
  endif()
  foreach(arch ${FAST_GICP_CUDA_ARCH})
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${arch},code=sm_${arch}")
  endforeach()
  # New FastGICPCuda class is always built when BUILD_VGICP_CUDA=ON.
  add_definitions(-DUSE_FAST_GICP_CUDA)
  set(FAST_GICP_CUVS_LIBS "")
  if(USE_CUVS)
    # cuVS install layout: header `cuvs/neighbors/brute_force.h` + libcuvs.so.
    # Recommended install via the rapidsai conda channel:
    #   mamba install -c rapidsai -c conda-forge cuvs-cu12
    find_path(CUVS_INCLUDE_DIR cuvs/neighbors/brute_force.h
      HINTS $ENV{CONDA_PREFIX}/include /usr/include /usr/local/include
      DOC "Directory containing cuvs/neighbors/brute_force.h")
    find_library(CUVS_LIBRARY cuvs
      HINTS $ENV{CONDA_PREFIX}/lib /usr/lib /usr/local/lib
      DOC "libcuvs.so")
    if(NOT CUVS_INCLUDE_DIR OR NOT CUVS_LIBRARY)
      message(FATAL_ERROR
        "USE_CUVS=ON but cuVS was not found. "
        "Install with `mamba install -c rapidsai -c conda-forge cuvs-cu12` "
        "or pass -DCUVS_INCLUDE_DIR=<path> -DCUVS_LIBRARY=<libcuvs.so>.")
    endif()
    message(STATUS "Found cuVS: ${CUVS_LIBRARY} (include ${CUVS_INCLUDE_DIR})")
    include_directories(${CUVS_INCLUDE_DIR})
    list(APPEND FAST_GICP_CUVS_LIBS ${CUVS_LIBRARY})
    add_definitions(-DUSE_CUVS)
  endif()

  # ---- new FastGICPCuda library (clean, no nvbio) ----
  # Built as a separate target so that pre-existing nvbio/VGICP issues under
  # newer CUDA toolchains don't block the new classic-GICP CUDA path.
  # B1: only the FastGICPCuda skeleton. Brute-force KNN will be added in B2
  # with a nvbio-free implementation suitable for source < 100k & target up to
  # several hundred K (heap-of-K maintained per source thread).
  cuda_add_library(fast_gicp_cuda_kernels SHARED
    src/fast_gicp/cuda/fast_gicp_cuda_core.cu
  )
  if(USE_CUVS)
    target_link_libraries(fast_gicp_cuda_kernels ${FAST_GICP_CUVS_LIBS})
  endif()
  target_include_directories(fast_gicp_cuda_kernels PRIVATE
    include
    thirdparty/Eigen
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )
  cuda_add_cublas_to_target(fast_gicp_cuda_kernels)

  target_sources(fast_gicp PRIVATE
    src/fast_gicp/gicp/fast_gicp_cuda.cpp
  )
  target_link_libraries(fast_gicp
    fast_gicp_cuda_kernels
  )
  add_dependencies(fast_gicp fast_gicp_cuda_kernels)

  # ---- legacy fast_vgicp_cuda (voxel VGICP + NDT) ----
  # This library uses nvbio and may fail to build on newer CUDA toolchains.
  # Gated by FAST_GICP_BUILD_LEGACY_VGICP_CUDA so users can opt-in.
  option(FAST_GICP_BUILD_LEGACY_VGICP_CUDA "Build legacy nvbio-based VGICP/NDT CUDA library" OFF)
  if(FAST_GICP_BUILD_LEGACY_VGICP_CUDA)
    add_definitions(-DUSE_VGICP_CUDA)
    cuda_add_library(fast_vgicp_cuda SHARED
      src/fast_gicp/cuda/fast_vgicp_cuda.cu
      src/fast_gicp/cuda/brute_force_knn.cu
      src/fast_gicp/cuda/covariance_estimation.cu
      src/fast_gicp/cuda/covariance_estimation_rbf.cu
      src/fast_gicp/cuda/covariance_regularization.cu
      src/fast_gicp/cuda/gaussian_voxelmap.cu
      src/fast_gicp/cuda/find_voxel_correspondences.cu
      src/fast_gicp/cuda/compute_derivatives.cu
      src/fast_gicp/cuda/compute_mahalanobis.cu
      src/fast_gicp/cuda/ndt_cuda.cu
      src/fast_gicp/cuda/ndt_compute_derivatives.cu
    )
    target_include_directories(fast_vgicp_cuda PRIVATE
      include
      thirdparty/Eigen
      thirdparty/nvbio
      ${catkin_INCLUDE_DIRS}
    )
    target_link_libraries(fast_vgicp_cuda ${catkin_LIBRARIES})
    cuda_add_cublas_to_target(fast_vgicp_cuda)
    target_sources(fast_gicp PRIVATE
      src/fast_gicp/gicp/fast_vgicp_cuda.cpp
      src/fast_gicp/ndt/ndt_cuda.cpp
    )
    target_link_libraries(fast_gicp fast_vgicp_cuda)
    add_dependencies(fast_gicp fast_vgicp_cuda)
    if(catkin_FOUND)
      install(TARGETS fast_vgicp_cuda
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
    endif()
  endif()
endif()

### TEST ###
if(BUILD_test)
  add_executable(gicp_selfcheck src/test/gicp_selfcheck.cpp)
  add_dependencies(gicp_selfcheck fast_gicp)
  target_link_libraries(gicp_selfcheck ${PCL_LIBRARIES} fast_gicp)

  add_executable(dynamic_rejection_test src/test/dynamic_rejection_test.cpp)
  add_dependencies(dynamic_rejection_test fast_gicp)
  target_link_libraries(dynamic_rejection_test fast_gicp)
endif()

if(BUILD_benchmarks)
  add_executable(gicp_benchmark src/test/gicp_benchmark.cpp)
  add_dependencies(gicp_benchmark fast_gicp)
  target_link_libraries(gicp_benchmark ${PCL_LIBRARIES} fast_gicp)
endif()

if(catkin_FOUND)
  ###################################
  ## catkin specific configuration ##
  ###################################
  install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

  install(DIRECTORY include/
    DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.hpp")
elseif (ament_cmake_FOUND)
  ##################################
  ## ament specific configuration ##
  ##################################
  ament_export_include_directories(include)
  ament_export_libraries(fast_gicp)
  ament_package()

  install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib)

  install(
      DIRECTORY "include/"
      DESTINATION include
    )
endif()
