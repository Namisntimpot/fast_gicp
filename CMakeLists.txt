cmake_minimum_required(VERSION 3.15.0)
project(fast_gicp)

# Headless-server friendly defaults: nothing that needs a display is built
# unless explicitly requested. The apps (gicp_align / gicp_kitti) link PCL
# visualization paths and need X at runtime, so they default to OFF.
option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" OFF)
option(BUILD_apps "Build application programs (need a display at runtime)" OFF)
option(BUILD_test "Build test programs" OFF)
option(BUILD_benchmarks "Build benchmark programs" OFF)
option(BUILD_PYTHON_BINDINGS "Build python bindings" OFF)

# Version string baked into the python module as pygicp.__version__
# (set by setup.py to "<base>+<git-branch>.<git-sha>[.dirty]").
set(FAST_GICP_VERSION_INFO "" CACHE STRING
  "Version string embedded into pygicp.__version__")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

#############################################
## Dependency search: conda first, then OS ##
#############################################
# Every find_package() below searches the active conda environment first
# (CMAKE_PREFIX_PATH front), then falls back to the default system locations,
# and finally fails with an actionable error message.
set(FAST_GICP_CONDA_PREFIX "")
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  set(FAST_GICP_CONDA_PREFIX "$ENV{CONDA_PREFIX}")
  list(PREPEND CMAKE_PREFIX_PATH "${FAST_GICP_CONDA_PREFIX}")
  if(NOT DEFINED BOOST_ROOT AND EXISTS "${FAST_GICP_CONDA_PREFIX}/include/boost")
    set(BOOST_ROOT "${FAST_GICP_CONDA_PREFIX}")
  endif()
  message(STATUS "fast_gicp: preferring dependencies from conda env: ${FAST_GICP_CONDA_PREFIX}")
else()
  message(STATUS "fast_gicp: no conda env active, using system dependencies")
endif()

find_package(PCL QUIET COMPONENTS common io filters search registration)
if(NOT PCL_FOUND)
  message(FATAL_ERROR
    "fast_gicp: PCL was not found.\n"
    "Searched the active conda env first (CONDA_PREFIX='${FAST_GICP_CONDA_PREFIX}'), then system paths.\n"
    "Install one of:\n"
    "  conda/mamba install -c conda-forge pcl boost-cpp eigen\n"
    "  sudo apt install libpcl-dev\n"
    "or pass -DPCL_DIR=<dir containing PCLConfig.cmake>.")
endif()
message(STATUS "fast_gicp: PCL ${PCL_VERSION} (${PCL_DIR})")
add_definitions(${PCL_DEFINITIONS})

find_package(Eigen3 QUIET)
if(TARGET Eigen3::Eigen AND NOT EIGEN3_INCLUDE_DIR)
  get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
endif()
if(NOT EIGEN3_INCLUDE_DIR)
  # Manual fallback: conda env include dir, then standard system locations.
  foreach(_eigen_cand
      "${FAST_GICP_CONDA_PREFIX}/include/eigen3"
      "/usr/include/eigen3"
      "/usr/local/include/eigen3")
    if(NOT EIGEN3_INCLUDE_DIR AND EXISTS "${_eigen_cand}/Eigen/Core")
      set(EIGEN3_INCLUDE_DIR "${_eigen_cand}")
    endif()
  endforeach()
endif()
if(NOT EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR
    "fast_gicp: Eigen3 was not found (conda env first, then system).\n"
    "Install one of:\n"
    "  conda/mamba install -c conda-forge eigen\n"
    "  sudo apt install libeigen3-dev\n"
    "or pass -DEIGEN3_INCLUDE_DIR=<dir containing Eigen/Core>.")
endif()
message(STATUS "fast_gicp: Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")

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
  # CUDA toolkit lookup order: FAST_GICP_CUDA_ROOT env var, conda env
  # (only when its layout is complete enough for FindCUDA: nvcc + headers
  # in <root>/include), then FindCUDA's own defaults (nvcc on PATH,
  # /usr/local/cuda). Each candidate that fails is discarded and the next
  # one is tried; only when all fail do we error out.
  set(_fast_gicp_cuda_roots "")
  if(DEFINED ENV{FAST_GICP_CUDA_ROOT} AND EXISTS "$ENV{FAST_GICP_CUDA_ROOT}/bin/nvcc")
    list(APPEND _fast_gicp_cuda_roots "$ENV{FAST_GICP_CUDA_ROOT}")
  endif()
  if(NOT "${FAST_GICP_CONDA_PREFIX}" STREQUAL ""
     AND EXISTS "${FAST_GICP_CONDA_PREFIX}/bin/nvcc"
     AND EXISTS "${FAST_GICP_CONDA_PREFIX}/include/cuda_runtime.h")
    # Only a complete conda toolkit layout; the split conda cuda packages
    # (headers under targets/<arch>/, nvcc without its profile) cannot be
    # driven by FindCUDA and are skipped in favour of the system toolkit.
    list(APPEND _fast_gicp_cuda_roots "${FAST_GICP_CONDA_PREFIX}")
  endif()
  if(EXISTS "/usr/local/cuda/bin/nvcc")
    list(APPEND _fast_gicp_cuda_roots "/usr/local/cuda")
  endif()
  list(APPEND _fast_gicp_cuda_roots "DEFAULT")
  list(GET FAST_GICP_CUDA_ARCH 0 _fast_gicp_smoke_arch)
  foreach(_cuda_root ${_fast_gicp_cuda_roots})
    if(NOT CUDA_FOUND)
      foreach(_v CUDA_TOOLKIT_ROOT_DIR CUDA_NVCC_EXECUTABLE CUDA_TOOLKIT_INCLUDE
                 CUDA_CUDART_LIBRARY CUDA_CUDA_LIBRARY CUDA_VERSION
                 CUDA_cublas_LIBRARY CUDA_cufft_LIBRARY CUDA_cupti_LIBRARY
                 CUDA_curand_LIBRARY CUDA_cusolver_LIBRARY CUDA_cusparse_LIBRARY)
        unset(${_v} CACHE)
        unset(${_v})
      endforeach()
      if(NOT "${_cuda_root}" STREQUAL "DEFAULT")
        set(CUDA_TOOLKIT_ROOT_DIR "${_cuda_root}")
        message(STATUS "fast_gicp: trying CUDA toolkit at ${_cuda_root}")
      else()
        message(STATUS "fast_gicp: trying default CUDA toolkit search (PATH, /usr/local/cuda)")
      endif()
      find_package(CUDA QUIET)
      if(CUDA_FOUND)
        # Smoke test: FindCUDA accepting a root does not guarantee nvcc can
        # actually compile (e.g. conda's split packaging loses cicc), nor
        # that it supports the requested arch. Reject broken candidates.
        set(_smoke_src "${CMAKE_BINARY_DIR}/fast_gicp_nvcc_smoke.cu")
        if(NOT EXISTS "${_smoke_src}")
          file(WRITE "${_smoke_src}"
            "__global__ void fast_gicp_smoke_kernel() {}\nint main() { return 0; }\n")
        endif()
        execute_process(
          COMMAND "${CUDA_NVCC_EXECUTABLE}" -std=c++17
                  -gencode=arch=compute_${_fast_gicp_smoke_arch},code=sm_${_fast_gicp_smoke_arch}
                  -o "${CMAKE_BINARY_DIR}/fast_gicp_nvcc_smoke.o" -c "${_smoke_src}"
          RESULT_VARIABLE _smoke_rc
          OUTPUT_VARIABLE _smoke_out
          ERROR_VARIABLE _smoke_err)
        if(NOT _smoke_rc EQUAL 0)
          message(STATUS
            "fast_gicp: rejecting CUDA at ${CUDA_TOOLKIT_ROOT_DIR} -- nvcc "
            "smoke test failed (sm_${_fast_gicp_smoke_arch}): ${_smoke_err}")
          set(CUDA_FOUND FALSE)
        endif()
      endif()
    endif()
  endforeach()
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR
      "fast_gicp: BUILD_VGICP_CUDA=ON but no usable CUDA toolkit was found "
      "(checked FAST_GICP_CUDA_ROOT, conda env, PATH, /usr/local/cuda).\n"
      "Either install a CUDA toolkit, point FAST_GICP_CUDA_ROOT at one, "
      "or build CPU-only with FAST_GICP_BUILD_VGICP_CUDA=0.")
  endif()
  message(STATUS "fast_gicp: CUDA ${CUDA_VERSION} (${CUDA_TOOLKIT_ROOT_DIR})")
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
  thirdparty/nanoflann/include
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
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    message(FATAL_ERROR
      "fast_gicp: pybind11 was not found (conda env first, then system).\n"
      "Install one of:\n"
      "  pip install pybind11\n"
      "  conda/mamba install -c conda-forge pybind11\n"
      "or pass -Dpybind11_DIR=<dir containing pybind11Config.cmake>.")
  endif()
  pybind11_add_module(pygicp
    src/python/main.cpp
  )
  target_include_directories(pygicp PUBLIC
    include
    thirdparty/nanoflann/include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )
  target_link_libraries(pygicp PRIVATE
    fast_gicp
  )
  if(NOT "${FAST_GICP_VERSION_INFO}" STREQUAL "")
    # Exposed as pygicp.__version__ (see src/python/main.cpp).
    target_compile_definitions(pygicp PRIVATE
      VERSION_INFO=${FAST_GICP_VERSION_INFO})
  endif()
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
    # The C wrapper symbols live in libcuvs_c.so; libcuvs.so holds the C++
    # implementation. Linking both keeps RPATH resolution simple.
    find_library(CUVS_C_LIBRARY cuvs_c
      HINTS $ENV{CONDA_PREFIX}/lib /usr/lib /usr/local/lib
      DOC "libcuvs_c.so (C API wrapper)")
    find_library(CUVS_LIBRARY cuvs
      HINTS $ENV{CONDA_PREFIX}/lib /usr/lib /usr/local/lib
      DOC "libcuvs.so (C++ implementation)")
    if(NOT CUVS_INCLUDE_DIR OR NOT CUVS_C_LIBRARY OR NOT CUVS_LIBRARY)
      message(FATAL_ERROR
        "USE_CUVS=ON but cuVS was not found. "
        "Install with `mamba install -c rapidsai -c conda-forge libcuvs cuvs cuda-version=12.8` "
        "or pass -DCUVS_INCLUDE_DIR=<path> -DCUVS_C_LIBRARY=<libcuvs_c.so> "
        "-DCUVS_LIBRARY=<libcuvs.so>.")
    endif()
    message(STATUS "Found cuVS: ${CUVS_C_LIBRARY} (include ${CUVS_INCLUDE_DIR})")
    include_directories(${CUVS_INCLUDE_DIR})
    list(APPEND FAST_GICP_CUVS_LIBS ${CUVS_C_LIBRARY} ${CUVS_LIBRARY})
    add_definitions(-DUSE_CUVS)
  endif()

  # ---- new FastGICPCuda library (clean, no nvbio) ----
  # Built as a separate target so that pre-existing nvbio/VGICP issues under
  # newer CUDA toolchains don't block the new classic-GICP CUDA path.
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
