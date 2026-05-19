# -*- coding: utf-8 -*-
import os
import sys
import glob
import shutil
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # cfg = "Debug" if self.debug else "Release"
        cfg = "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.

        # if some dependencies are installed in conda env...
        conda_prefix = os.environ.get("CONDA_PREFIX", None)
        pybind11_cmake_dir = None
        try:
            import pybind11
            pybind11_cmake_dir = pybind11.get_cmake_dir()
        except Exception:
            pybind11_cmake_dir = None

        rpath_entries = ["$ORIGIN"]
        if conda_prefix is not None:
            rpath_entries.append(f"{conda_prefix}/lib")

        # Feature defaults for this branch:
        # - FastGICP dynamic-rejection (GNC + IRLS) is always compiled in.
        # - FastGICPCuda + brute-force KNN are ON by default (BUILD_VGICP_CUDA=ON).
        # - cuVS KNN backend stays OFF (heavy optional dep). Enable with
        #     FAST_GICP_USE_CUVS=1 in the environment.
        # - Legacy nvbio voxel-CUDA library (FastVGICPCuda / NDTCuda) stays OFF;
        #   it does not compile cleanly under CUDA 12.x because of a
        #   thrust::pair ambiguity. Enable with FAST_GICP_LEGACY_VGICP_CUDA=1.
        # - The optional test/benchmark binaries stay OFF.
        # All defaults can be overridden by setting the matching env var to 0/1.
        def _envflag(name, default):
            val = os.environ.get(name)
            if val is None:
                return default
            return val.strip().lower() in ("1", "on", "true", "yes")

        build_vgicp_cuda = _envflag("FAST_GICP_BUILD_VGICP_CUDA", True)
        use_cuvs = _envflag("FAST_GICP_USE_CUVS", False)
        legacy_vgicp = _envflag("FAST_GICP_LEGACY_VGICP_CUDA", False)
        build_apps = _envflag("FAST_GICP_BUILD_APPS", False)
        build_test = _envflag("FAST_GICP_BUILD_TEST", False)
        build_benchmarks = _envflag("FAST_GICP_BUILD_BENCHMARKS", False)
        cuda_arch = os.environ.get("FAST_GICP_CUDA_ARCH", "120")

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DEXAMPLE_VERSION_INFO={}".format(self.distribution.get_version()),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm,
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DBUILD_apps={}".format("ON" if build_apps else "OFF"),
            "-DBUILD_test={}".format("ON" if build_test else "OFF"),
            "-DBUILD_benchmarks={}".format("ON" if build_benchmarks else "OFF"),
            "-DBUILD_VGICP_CUDA={}".format("ON" if build_vgicp_cuda else "OFF"),
            "-DUSE_CUVS={}".format("ON" if use_cuvs else "OFF"),
            "-DFAST_GICP_BUILD_LEGACY_VGICP_CUDA={}".format(
                "ON" if legacy_vgicp else "OFF"),
            "-DFAST_GICP_CUDA_ARCH={}".format(cuda_arch),
            "-DCMAKE_BUILD_RPATH={}".format(";".join(rpath_entries)),
            "-DCMAKE_INSTALL_RPATH={}".format(";".join(rpath_entries)),
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        ]
        # The new fast_gicp_cuda_kernels SHARED library needs to be next to
        # the pygicp module so the runtime loader finds it via $ORIGIN.
        # extdir already contains "$ORIGIN" in the rpath; the install hook
        # at the bottom of build_extension also copies libfast_gicp*.so into
        # the source tree, and we extend it below to include the new lib.
        if conda_prefix is not None:
            cmake_args += [
                f"-DPCL_DIR={conda_prefix}/share/pcl-1.14",
                f"-DBOOST_ROOT={conda_prefix}",
            ]
        if pybind11_cmake_dir is not None:
            cmake_args.append(f"-Dpybind11_DIR={pybind11_cmake_dir}")
        build_args = []

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                pass
                # cmake_args += ["-GNinja"]

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
        # Copy all of the produced shared libraries next to pygicp so the
        # runtime loader can resolve them via the $ORIGIN rpath.
        # When BUILD_VGICP_CUDA=ON this includes libfast_gicp_cuda_kernels.so.
        patterns = ["libfast_gicp*.so*", "libfast_vgicp_cuda*.so*"]
        for pattern in patterns:
            for lib_path in glob.glob(os.path.join(extdir, pattern)):
                shutil.copy2(lib_path, ext.sourcedir)

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="pygicp",
    version="0.0.1",
    author="k.koide",
    author_email="k.koide@aist.go.jp",
    description="A collection of GICP-based point cloud registration algorithms",
    long_description="",
    ext_modules=[CMakeExtension("pygicp")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
