# -*- coding: utf-8 -*-
"""Build/install script for pygicp (the fast_gicp python bindings). Linux only.

Supported install methods (both leave exactly ONE active pygicp in the
current environment -- any previous install visible from this interpreter,
including editable hooks, copied .so files, dist-info/egg-info leftovers and
stale copies in ~/.local/lib/pythonX.Y/site-packages, is purged first):

    pip install -e .                  # editable install (recommended for dev)
    pip install .                     # regular install
    python setup.py install           # direct install into the active env
    python setup.py install --user    # direct install into the user site
    python setup.py uninstall         # remove every pygicp visible from here

Dependency lookup order: active conda environment first, then system
locations, then a hard error with installation hints (raised by CMake).

Headless-server friendly: nothing display-dependent is built by default
(BUILD_apps=OFF). Feature toggles via environment variables (0/1):

    FAST_GICP_BUILD_VGICP_CUDA   default 1   FastGICPCuda + CUDA kernels
    FAST_GICP_USE_CUVS           default 0   cuVS KNN backend (heavy dep)
    FAST_GICP_LEGACY_VGICP_CUDA  default 0   legacy nvbio VGICP/NDT CUDA lib
    FAST_GICP_BUILD_APPS         default 0   gicp_align/gicp_kitti (need X)
    FAST_GICP_BUILD_TEST         default 0   C++ test binaries
    FAST_GICP_BUILD_BENCHMARKS   default 0   C++ benchmark binaries
    FAST_GICP_CUDA_ARCH          default "120"  semicolon-separated sm archs
    FAST_GICP_CUDA_ROOT          (unset)     explicit CUDA toolkit root

The installed version is stamped with the git branch + commit, e.g.
0.0.1+feature.dynamic.outlier.rejection.e0fd216, so you can always check
which branch/commit the environment is actually using:

    python -c "import pygicp; print(pygicp.__version__, pygicp.__file__)"
"""
import glob
import hashlib
import base64
import json
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import tempfile

from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install as _install_base

ROOT = os.path.abspath(os.path.dirname(__file__))
PKG = "pygicp"
BASE_VERSION = "0.0.1"


# --------------------------------------------------------------------------
# version stamping: <base>+<branch>.<sha>[.dirty]
# --------------------------------------------------------------------------
def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def compute_version():
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or ""
    sha = _git("rev-parse", "--short", "HEAD") or ""
    dirty = bool(_git("status", "--porcelain", "-uno") or "")
    parts = []
    if branch and branch != "HEAD":
        # sanitize to a valid PEP440 local-version segment
        parts.append(re.sub(r"[^a-zA-Z0-9.]+", ".", branch).strip(".").lower())
    if sha:
        parts.append(sha)
    if dirty:
        parts.append("dirty")
    if not parts:
        return BASE_VERSION
    return BASE_VERSION + "+" + ".".join(p for p in parts if p)


VERSION = compute_version()


# --------------------------------------------------------------------------
# environment helpers
# --------------------------------------------------------------------------
def _envflag(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "on", "true", "yes")


def _under_pip():
    """True when this setup.py runs inside a pip-driven (PEP 517) build."""
    return any(k in os.environ for k in (
        "PIP_BUILD_TRACKER", "PEP517_BUILD_BACKEND",
        "_PYPROJECT_HOOKS_BUILD_BACKEND"))


def effective_conda_prefix():
    """Conda env we are installing INTO, derived from the interpreter.

    Using sys.prefix (not $CONDA_PREFIX) makes things work even when the
    user runs `.../envs/foo/bin/pip install -e .` without activating foo.
    """
    prefix = os.path.realpath(sys.prefix)
    if os.path.isdir(os.path.join(prefix, "conda-meta")):
        env_var = os.environ.get("CONDA_PREFIX")
        if env_var and os.path.realpath(env_var) != prefix:
            print(f"[fast_gicp] WARNING: $CONDA_PREFIX={env_var} but the "
                  f"running interpreter belongs to {prefix}; using the "
                  f"interpreter's env for dependency lookup.")
        return prefix
    return os.environ.get("CONDA_PREFIX") or None


def _pybind11_cmake_dir():
    """conda/active env first, then system; None lets CMake try on its own.

    The persistent environment is preferred over `import pybind11` because
    under pip's build isolation the import resolves to a temporary overlay
    that vanishes after the install, leaving a dangling path in CMakeCache.
    """
    candidates = []
    conda_prefix = effective_conda_prefix()
    if conda_prefix:
        py = "python{}.{}".format(*sys.version_info[:2])
        candidates.append(os.path.join(
            conda_prefix, "lib", py, "site-packages", "pybind11",
            "share", "cmake", "pybind11"))
        candidates.append(os.path.join(
            conda_prefix, "share", "cmake", "pybind11"))
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "pybind11Config.cmake")):
            return cand
    try:
        import pybind11
        return pybind11.get_cmake_dir()
    except ImportError:
        return None


# --------------------------------------------------------------------------
# purge of previous installs (the "only one pygicp at a time" guarantee)
# --------------------------------------------------------------------------
def _candidate_install_dirs():
    dirs = []
    try:
        dirs += list(site.getsitepackages())
    except Exception:
        pass
    try:
        u = site.getusersitepackages()
        if u:
            dirs.append(u)
    except Exception:
        pass
    for key in ("purelib", "platlib"):
        try:
            dirs.append(sysconfig.get_paths()[key])
        except Exception:
            pass
    out = []
    for d in dirs:
        d = os.path.abspath(d)
        if d not in out and os.path.isdir(d):
            out.append(d)
    return out


def _canonical_dist_paths():
    """Metadata dir of the pygicp install that pip currently considers
    canonical. We leave it for pip to uninstall (deleting it mid-transaction
    can confuse pip); everything else is fair game."""
    paths = set()
    try:
        from importlib.metadata import distribution
        p = getattr(distribution(PKG), "_path", None)
        if p:
            paths.add(os.path.abspath(str(p)))
    except Exception:
        pass
    return paths


_PURGE_PATTERNS = [
    "pygicp.so", "pygicp.*.so",
    "libfast_gicp*.so*", "libfast_vgicp_cuda*.so*",
    "pygicp-*.dist-info", "pygicp-*.egg-info", "pygicp-*.egg",
    "pygicp.egg-link",
    "__editable__.pygicp-*.pth", "__editable___pygicp_*.py",
    os.path.join("__pycache__", "__editable___pygicp_*"),
]


def _remove_path(path, removed):
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        removed.append(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"[fast_gicp] WARNING: could not remove {path}: {exc}")


def purge_previous_installs(skip_canonical=False):
    """Remove every trace of older pygicp installs visible to this
    interpreter (env site-packages, user site, leftover editable hooks,
    easy-install.pth entries)."""
    skip = _canonical_dist_paths() if skip_canonical else set()
    removed = []
    for d in _candidate_install_dirs():
        for pat in _PURGE_PATTERNS:
            for path in glob.glob(os.path.join(d, pat)):
                if os.path.abspath(path) in skip:
                    continue
                _remove_path(path, removed)
        # scrub legacy `setup.py develop` entries pointing at fast_gicp clones
        pth = os.path.join(d, "easy-install.pth")
        if os.path.isfile(pth):
            try:
                with open(pth) as f:
                    lines = f.readlines()
                kept = [ln for ln in lines if "fast_gicp" not in ln]
                if kept != lines:
                    with open(pth, "w") as f:
                        f.writelines(kept)
                    removed.append(pth + " (fast_gicp lines)")
            except OSError as exc:
                print(f"[fast_gicp] WARNING: could not edit {pth}: {exc}")
    if removed:
        print("[fast_gicp] purged previous install artifacts:")
        for r in removed:
            print(f"[fast_gicp]   - {r}")
    return removed


def purge_in_tree_artifacts():
    """Stale in-tree .so files (possibly from another python version/env)
    shadow installed copies whenever cwd is the repo -- always start clean."""
    for pat in ("pygicp*.so", "libfast_gicp*.so*", "libfast_vgicp_cuda*.so*"):
        for path in glob.glob(os.path.join(ROOT, pat)):
            try:
                os.remove(path)
                print(f"[fast_gicp] removed stale in-tree artifact: {path}")
            except OSError:
                pass


def _pip_uninstall_all():
    """Best-effort: let pip cleanly remove whatever it knows about."""
    for _ in range(4):
        try:
            out = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", PKG],
                capture_output=True, text=True, cwd=tempfile.gettempdir())
        except Exception:
            return
        if out.returncode != 0 or "as it is not installed" in (out.stdout + out.stderr):
            return


# --------------------------------------------------------------------------
# CMake build
# --------------------------------------------------------------------------
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        editable = bool(self.inplace or getattr(self, "editable_mode", False))
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        conda_prefix = effective_conda_prefix()

        # Clean before building: stale in-tree .so always; previous installs
        # only when an actual install is in flight (pip drives build_ext for
        # both `pip install .` and `pip install -e .`; the direct
        # `setup.py install` path purges inside the install command itself).
        purge_in_tree_artifacts()
        if _under_pip():
            purge_previous_installs(skip_canonical=True)

        # Feature flags (see module docstring).
        build_vgicp_cuda = _envflag("FAST_GICP_BUILD_VGICP_CUDA", True)
        use_cuvs = _envflag("FAST_GICP_USE_CUVS", False)
        legacy_vgicp = _envflag("FAST_GICP_LEGACY_VGICP_CUDA", False)
        build_apps = _envflag("FAST_GICP_BUILD_APPS", False)
        build_test = _envflag("FAST_GICP_BUILD_TEST", False)
        build_benchmarks = _envflag("FAST_GICP_BUILD_BENCHMARKS", False)
        cuda_arch = os.environ.get("FAST_GICP_CUDA_ARCH", "120")
        cuda_root = os.environ.get("FAST_GICP_CUDA_ROOT", "")

        # Dedicated build dir per (python version, environment) so switching
        # envs can never reuse a CMake cache configured for another one.
        env_name = os.path.basename(conda_prefix) if conda_prefix else "system"
        pytag = "cp{}{}".format(*sys.version_info[:2])
        self.build_temp = os.path.join("build", f"cmake-{pytag}-{env_name}")
        stamp = {
            "python": os.path.realpath(sys.executable),
            "python_version": sys.version.split()[0],
            "conda_prefix": conda_prefix,
            "build_vgicp_cuda": build_vgicp_cuda,
            "use_cuvs": use_cuvs,
            "legacy_vgicp": legacy_vgicp,
            "cuda_arch": cuda_arch,
            "cuda_root": cuda_root,
        }
        self._refresh_build_temp(stamp)

        rpath_entries = ["$ORIGIN"]
        if conda_prefix:
            rpath_entries.append(os.path.join(conda_prefix, "lib"))

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DFAST_GICP_VERSION_INFO={VERSION}",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DBUILD_apps={}".format("ON" if build_apps else "OFF"),
            "-DBUILD_test={}".format("ON" if build_test else "OFF"),
            "-DBUILD_benchmarks={}".format("ON" if build_benchmarks else "OFF"),
            "-DBUILD_VGICP_CUDA={}".format("ON" if build_vgicp_cuda else "OFF"),
            "-DUSE_CUVS={}".format("ON" if use_cuvs else "OFF"),
            "-DFAST_GICP_BUILD_LEGACY_VGICP_CUDA={}".format(
                "ON" if legacy_vgicp else "OFF"),
            f"-DFAST_GICP_CUDA_ARCH={cuda_arch}",
            "-DCMAKE_BUILD_RPATH={}".format(";".join(rpath_entries)),
            "-DCMAKE_INSTALL_RPATH={}".format(";".join(rpath_entries)),
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
            "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON",
        ]
        pybind11_dir = _pybind11_cmake_dir()
        if pybind11_dir:
            cmake_args.append(f"-Dpybind11_DIR={pybind11_dir}")

        # CMake reads $CONDA_PREFIX for conda-first dependency lookup; pin it
        # to the env of the running interpreter (see effective_conda_prefix).
        cmake_env = os.environ.copy()
        if conda_prefix:
            cmake_env["CONDA_PREFIX"] = conda_prefix
        else:
            cmake_env.pop("CONDA_PREFIX", None)

        jobs = self.parallel or min(os.cpu_count() or 4, 32)
        build_args = ["--", f"-j{jobs}"]

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=cmake_env)
        subprocess.check_call(["cmake", "--build", "."] + build_args,
                              cwd=self.build_temp, env=cmake_env)

        if editable:
            # setuptools' editable mode copies only the pygicp extension back
            # into the source tree; the companion CMake libraries must follow
            # it there (resolved via the $ORIGIN rpath), or the temp build-lib
            # dir takes them to the grave.
            for pattern in ("pygicp*.so", "libfast_gicp*.so*",
                            "libfast_vgicp_cuda*.so*"):
                for lib in glob.glob(os.path.join(extdir, pattern)):
                    shutil.copy2(lib, ext.sourcedir)

        print(f"[fast_gicp] built {PKG} {VERSION}")
        print(f"[fast_gicp]   python      : {sys.executable}")
        print(f"[fast_gicp]   environment : {conda_prefix or 'system python'}")
        print(f"[fast_gicp]   CUDA        : {'ON' if build_vgicp_cuda else 'OFF'}"
              f"{' (arch ' + cuda_arch + ')' if build_vgicp_cuda else ''}")
        print(f"[fast_gicp]   editable    : {editable}")
        print(f"[fast_gicp]   artifacts   : {extdir}")

    def _refresh_build_temp(self, stamp):
        stamp_file = os.path.join(self.build_temp, "fast_gicp_build_stamp.json")
        if os.path.isdir(self.build_temp):
            old = None
            try:
                with open(stamp_file) as f:
                    old = json.load(f)
            except Exception:
                pass
            if old != stamp:
                print(f"[fast_gicp] build configuration changed; wiping "
                      f"{self.build_temp}")
                shutil.rmtree(self.build_temp, ignore_errors=True)
        os.makedirs(self.build_temp, exist_ok=True)
        with open(stamp_file, "w") as f:
            json.dump(stamp, f, indent=2)


# --------------------------------------------------------------------------
# `setup.py install [--user]` -- reimplemented because setuptools >= 80
# removed the stock install command. Produces a pip-uninstallable layout
# (proper .dist-info with RECORD).
# --------------------------------------------------------------------------
class CMakeInstall(_install_base):
    user_options = _install_base.user_options + [
        ("skip-verify", None, "skip the post-install import check"),
    ]
    boolean_options = _install_base.boolean_options + ["skip-verify"]

    def initialize_options(self):
        super().initialize_options()
        self.skip_verify = False

    def run(self):
        if self.root:
            # bdist_wheel drives `install` with --root into a temp tree;
            # keep the stock behaviour for that path.
            return super().run()
        self._direct_install()

    def _direct_install(self):
        print(f"[fast_gicp] direct install of {PKG} {VERSION} "
              f"({'user site' if self.user else 'environment site-packages'})")
        # 1) remove whatever pip knows about, then sweep all leftovers
        _pip_uninstall_all()
        purge_previous_installs()
        purge_in_tree_artifacts()

        # 2) build
        self.run_command("build_ext")
        bext = self.get_finalized_command("build_ext")
        ext_path = os.path.abspath(bext.get_ext_fullpath(PKG))
        extdir = os.path.dirname(ext_path)

        artifacts = sorted(
            set(glob.glob(os.path.join(extdir, "pygicp*.so")) +
                glob.glob(os.path.join(extdir, "libfast_gicp*.so*")) +
                glob.glob(os.path.join(extdir, "libfast_vgicp_cuda*.so*"))))
        if not any(os.path.basename(a).startswith("pygicp") for a in artifacts):
            raise RuntimeError(f"build produced no pygicp extension in {extdir}")

        # 3) target directory (install_lib already honours --user/--prefix)
        target = os.path.abspath(self.install_lib or
                                 (site.getusersitepackages() if self.user
                                  else sysconfig.get_paths()["platlib"]))
        os.makedirs(target, exist_ok=True)
        if not os.access(target, os.W_OK):
            raise RuntimeError(
                f"{target} is not writable; retry with "
                f"`python setup.py install --user`")
        if self.user:
            print("[fast_gicp] NOTE: the user site is shared by every conda "
                  "env with python {}.{} -- prefer `pip install -e .` per env "
                  "if that is not what you want.".format(*sys.version_info[:2]))

        # 4) copy artifacts + write dist-info so pip can see/uninstall it
        installed = []
        for a in artifacts:
            dst = os.path.join(target, os.path.basename(a))
            shutil.copy2(a, dst)
            installed.append(dst)
            print(f"[fast_gicp] installed {dst}")
        self._write_dist_info(target, installed)

        # 5) verify that a fresh interpreter picks up exactly this install
        if not self.skip_verify:
            self._verify(target)

    def _write_dist_info(self, target, installed_files):
        di = os.path.join(target, f"{PKG}-{VERSION}.dist-info")
        if os.path.isdir(di):
            shutil.rmtree(di)
        os.makedirs(di)

        def _write(name, content):
            path = os.path.join(di, name)
            with open(path, "w") as f:
                f.write(content)
            return path

        meta_files = [
            _write("METADATA",
                   "Metadata-Version: 2.1\n"
                   f"Name: {PKG}\n"
                   f"Version: {VERSION}\n"
                   "Summary: A collection of GICP-based point cloud "
                   "registration algorithms\n"),
            _write("INSTALLER", "fast_gicp-setup.py\n"),
            _write("top_level.txt", f"{PKG}\n"),
            _write("direct_url.json", json.dumps(
                {"url": "file://" + ROOT, "dir_info": {"editable": False}})),
        ]

        def _record_line(path):
            rel = os.path.relpath(path, target)
            with open(path, "rb") as f:
                data = f.read()
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()).rstrip(b"=").decode()
            return f"{rel},sha256={digest},{len(data)}"

        record_path = os.path.join(di, "RECORD")
        lines = [_record_line(p) for p in installed_files + meta_files]
        lines.append(f"{os.path.relpath(record_path, target)},,")
        with open(record_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[fast_gicp] wrote {di}")

    def _verify(self, target):
        code = ("import os, pygicp; "
                "print(pygicp.__version__); "
                "print(os.path.abspath(pygicp.__file__))")
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True,
                             cwd=tempfile.gettempdir())
        if out.returncode != 0:
            raise RuntimeError(
                f"post-install import check FAILED:\n{out.stderr}")
        got_version, got_file = out.stdout.strip().splitlines()[-2:]
        ok_ver = got_version == VERSION
        ok_loc = os.path.realpath(got_file).startswith(os.path.realpath(target))
        print(f"[fast_gicp] verify: import pygicp -> {got_file}")
        print(f"[fast_gicp] verify: __version__   -> {got_version}")
        if not (ok_ver and ok_loc):
            raise RuntimeError(
                "post-install check FAILED: another pygicp shadows this "
                f"install (expected version {VERSION} under {target}). "
                "Run `python setup.py uninstall` and install again.")
        print("[fast_gicp] verify: OK -- this environment now uses exactly "
              "this build")


class UninstallCommand(Command):
    description = "remove every pygicp installation visible from this interpreter"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        _pip_uninstall_all()
        purge_previous_installs()
        purge_in_tree_artifacts()
        print("[fast_gicp] uninstall complete")


setup(
    name=PKG,
    version=VERSION,
    author="k.koide",
    author_email="k.koide@aist.go.jp",
    description="A collection of GICP-based point cloud registration algorithms",
    long_description="",
    python_requires=">=3.8",
    ext_modules=[CMakeExtension(PKG)],
    cmdclass={
        "build_ext": CMakeBuild,
        "install": CMakeInstall,
        "uninstall": UninstallCommand,
    },
    zip_safe=False,
)
