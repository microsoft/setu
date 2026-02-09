import io
import os
import subprocess
import sys
from datetime import datetime
from shutil import which
from typing import List

import torch
from packaging.version import Version, parse
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools_scm import get_version
from torch.utils.cpp_extension import CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)

# Setu only supports Linux platform
assert sys.platform.startswith("linux"), "Setu only supports Linux platform ."


def is_git_repo(root_dir):
    """Check if the given directory is part of a git repository."""
    git_dir = os.path.join(root_dir, ".git")
    return os.path.exists(git_dir) and os.path.isdir(git_dir)


def get_build_directory(root_dir, debug=False, base_temp_dir=None):
    """
    Determine the appropriate build directory based on the installation context.

    Args:
        root_dir: Root directory of the project
        debug: Whether this is a debug build
        base_temp_dir: Default temp directory to use if not in a git repo

    Returns:
        Path to the build directory
    """
    # Determine build configuration
    default_cfg = "Debug" if debug else "RelWithDebInfo"
    cfg = os.getenv("CMAKE_BUILD_TYPE", default_cfg).lower()

    if is_git_repo(root_dir):
        # We're in a git repo (development mode)
        # Use project_root/build with subdirectories for different build types
        return os.path.join(root_dir, "build", cfg)
    else:
        # We're installing from a package (pip install)
        # Use the default temp directory to avoid polluting the package
        assert base_temp_dir, "base_temp_dir must be provided when not in a git repo"
        return base_temp_dir


def is_sccache_available() -> bool:
    return which("sccache") is not None


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_setu_version() -> str:
    version = get_version(
        write_to="setu/_version.py",  # TODO: move this to pyproject.toml
    )

    if _is_cuda():
        cuda_version = str(get_nvcc_cuda_version())
        cuda_version_str = cuda_version.replace(".", "")[:3]
        torch_version = torch.__version__.split("+")[0]
        torch_version = torch_version.split("-")[0]
        torch_version_str = torch_version.replace(".", "")[:3]

        pypi_release_cuda_version = os.getenv("PYPI_RELEASE_CUDA_VERSION", "")
        pypi_release_torch_version = os.getenv("PYPI_RELEASE_TORCH_VERSION", "")

        is_pypi_release = (
            pypi_release_cuda_version == cuda_version_str
            and pypi_release_torch_version == torch_version_str
        )

        is_nightly_build = os.getenv("IS_NIGHTLY_BUILD", "false") == "true"

        if is_nightly_build:
            # the version would be something like 0.0.2.dev17+g6833d6f
            # but for nightly builds, we want to keep the version as 0.0.2.dev{datetime}
            # remove part after dev and replace it with the current datetime
            version = (
                version.split("dev")[0] + f"dev{datetime.now().strftime('%Y%m%d%H')}"
            )

        # a version name can't have two "+" characters
        # so if the name already has a "+" character, we can add
        # the cuda and torch version as a suffix with "."
        # else we can add it with "+"
        sep = "." if "+" in version else "+"

        # skip this for source tarball, required for pypi
        if "sdist" not in sys.argv and not is_pypi_release:
            version += f"{sep}cu{cuda_version_str}torch{torch_version_str}"
    else:
        raise RuntimeError("Unknown runtime environment")

    return version


class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    #
    # Determine number of compilation jobs.
    #
    def compute_num_jobs(self):
        num_jobs = int(
            os.getenv("MAX_JOBS") or os.getenv("CMAKE_BUILD_PARALLEL_LEVEL") or 0
        )
        if not num_jobs:
            try:
                # os.sched_getaffinity() isn't universally available, so fall back
                # to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        return num_jobs

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension, build_dir: str) -> None:
        # Use custom build directory if provided, otherwise use default
        config_key = f"{ext.cmake_lists_dir}:{build_dir}"

        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if config_key in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[config_key] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        # Use Debug as default when in git repo for development, otherwise RelWithDebInfo
        if is_git_repo(ROOT_DIR):
            default_cfg = "Debug"  # Always Debug for git repo development
        else:
            default_cfg = (
                "Debug" if self.debug else "RelWithDebInfo"
            )  # RelWithDebInfo for packages
        cfg = os.getenv("CMAKE_BUILD_TYPE", default_cfg)

        # where .so files will be written, should be the same for all extensions
        # that use the same CMakeLists.txt.
        outdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(outdir),
        ]

        verbose = bool(int(os.getenv("VERBOSE", "0")))
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        if is_sccache_available():
            cmake_args += [
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
            ]
        elif is_ccache_available():
            cmake_args += [
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ["-DSETU_PYTHON_EXECUTABLE={}".format(sys.executable)]

        #
        # Setup build tool
        #
        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args], cwd=build_dir
        )

    def build_extensions(self) -> None:
        # Check if we should skip the C++ build entirely
        if os.getenv("SETU_SKIP_CPP_BUILD", "0") == "1":
            print("Skipping C++ build (SETU_SKIP_CPP_BUILD=1)")
            return

        # Standard setuptools build process
        # Determine the correct build directory without changing self.build_temp
        # This preserves other setuptools functionality while using our build directory structure
        build_directory = get_build_directory(
            ROOT_DIR, debug=self.debug, base_temp_dir=self.build_temp
        )

        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        os.makedirs(build_directory, exist_ok=True)

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext, build_directory)

            ext_target_name = remove_prefix(ext.name, "setu.")
            num_jobs = self.compute_num_jobs()

            build_args = [
                "--build",
                ".",
                "--target",
                ext_target_name,
                "-j",
                str(num_jobs),
            ]

            subprocess.check_call(["cmake", *build_args], cwd=build_directory)

    def copy_extensions_to_source(self):
        """Override to skip copying when we skipped the C++ build."""
        if os.getenv("SETU_SKIP_CPP_BUILD", "0") == "1":
            print("Skipping copy_extensions_to_source since C++ build was skipped")
            return
        # Standard setuptools behavior
        super().copy_extensions_to_source()


def _is_cuda() -> bool:
    return torch.version.cuda is not None


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output(
        [CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    if _is_cuda():
        with open(get_path("requirements.txt")) as f:
            requirements = f.read().strip().split("\n")
    else:
        raise ValueError("Unsupported platform, please use CUDA.")

    return requirements


def get_package_name() -> str:
    # nightly builds are published under the setu-nightly package
    if os.getenv("IS_NIGHTLY_BUILD", "false") == "true":
        return "setu-nightly"

    return "setu"


ext_modules = []

ext_modules.append(CMakeExtension(name="setu._commons"))
ext_modules.append(CMakeExtension(name="setu._client"))


setup(
    name=get_package_name(),
    author="Vajra Team",
    version=get_setu_version(),
    license="Apache 2.0",
    description=("A high-throughput and low-latency LLM inference system"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=("csrc")),
    python_requires=">=3.12",
    install_requires=get_requirements(),
    entry_points={
        "console_scripts": [
            "setu-cluster=setu.ray.__main__:main",
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": cmake_build_ext},
)
