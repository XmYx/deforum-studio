import os
import platform
import re
import sys
from distutils.core import Command
from setuptools import find_packages, setup

python_version = ".".join(map(str, sys.version_info[:2]))
python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
os_name = platform.system().lower()
machine = platform.machine().lower()

TORCH_VERSION = "2.12.0"
TORCHVISION_VERSION = "0.27.0"

# Supported values:
#   cu132, cu13.2, cuda13.2, cuda
#   cu130, cu13.0, cuda13.0
#   cu128, cu12.8, cuda12.8
#   rocm72, rocm7.2, rocm, amd
#   cpu
torch_backend = (
    os.environ.get("DEFORUM_TORCH_BACKEND", "cu132")
    .lower()
    .replace("-", "")
    .replace("_", "")
    .replace(".", "")
)

backend_aliases = {
    "cuda": "cu132",
    "cuda132": "cu132",
    "cu132": "cu132",
    "cuda130": "cu130",
    "cu130": "cu130",
    "cuda128": "cu128",
    "cu128": "cu128",
    "rocm": "rocm72",
    "rocm72": "rocm72",
    "amd": "rocm72",
    "cpu": "cpu",
}

torch_backend = backend_aliases.get(torch_backend, torch_backend)

supported_python_versions = {"3.10", "3.11", "3.12", "3.13", "3.14"}

if python_version not in supported_python_versions:
    sys.exit(
        f"Unsupported Python version: {python_version}. "
        f"Supported: {', '.join(sorted(supported_python_versions))}"
    )

if os_name == "linux":
    platform_tag = "manylinux_2_28_aarch64" if machine in {"aarch64", "arm64"} else "manylinux_2_28_x86_64"
elif os_name == "windows":
    platform_tag = "win_amd64"
elif os_name == "darwin":
    platform_tag = None
else:
    sys.exit(f"Unsupported OS: {os_name}")

torch_wheel_indexes = {
    "cu132": ("https://download.pytorch.org/whl/cu132", "cu132"),
    "cu130": ("https://download.pytorch.org/whl/cu130", "cu130"),
    "cu128": ("https://download.pytorch.org/whl/cu128", "cu128"),
    "rocm72": ("https://download.pytorch.org/whl/rocm7.2", "rocm7.2"),
}

if torch_backend == "rocm72" and os_name != "linux":
    sys.exit("ROCm 7.2 wheels are only supported on Linux. Use DEFORUM_TORCH_BACKEND=cu132 or cpu on this platform.")

if torch_backend.startswith("cu") and os_name not in {"linux", "windows"}:
    torch_backend = "cpu"

if torch_backend != "cpu" and torch_backend not in torch_wheel_indexes:
    sys.exit(
        f"Unsupported DEFORUM_TORCH_BACKEND={torch_backend!r}. "
        "Use cu132, cu130, cu128, rocm72, amd, or cpu."
    )


def torch_dep(package_name, version):
    if torch_backend == "cpu" or platform_tag is None:
        return f"{package_name}=={version}"

    wheel_index, local_version = torch_wheel_indexes[torch_backend]
    wheel_name = f"{package_name}-{version}%2B{local_version}-{python_tag}-{python_tag}-{platform_tag}.whl"
    return f"{package_name}@{wheel_index}/{wheel_name}"


# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/deforum/dependency_versions_table.py
_deps = [
    torch_dep("torch", TORCH_VERSION),
    torch_dep("torchvision", TORCHVISION_VERSION),
    "einops>=0.6.0",
    "numexpr>=2.8.4",
    "matplotlib>=3.7.1",
    "pandas>=1.5.3",
    "av>=10.0.0",
    "pims>=0.6.1",
    "imageio-ffmpeg>=0.4.8",
    "rich>=13.3.2",
    "gdown>=4.7.1",
    "py3d>=0.0.87",
    "librosa>=0.10.0.post2",
    "numpy==1.26.4",
    "opencv-python-headless",
    "timm>=0.6.13",
    "transformers>=4.40.2",
    "omegaconf>=2.3.0",
    "aiohttp>=3.9.3",
    "psutil>=5.9.6",
    "clip-interrogator>=0.6.0",
    "streamlit>=1.27.2",
    "torchsde>=0.2.5",
    "fastapi>=0.100.0",
    "diffusers==0.30.0",
    "accelerate>=0.29.3",
    "python-decouple>=3.8",
    "mutagen>=1.47.0",
    "imageio[ffmpeg]>=2.34.1",
    "xformers>=0.0.26.post1; python_version < '3.14' and platform_system != 'Darwin'",
    "tensorrt>=10.0.1; platform_system == 'Linux' and platform_machine == 'x86_64' and python_version < '3.14'",
    "onnx_graphsurgeon>=0.5.2",
    "onnx<=1.19.0",
    "zstandard>=0.22.0",
    "polygraphy>=0.49.9",
    "kornia>=0.7.2",
    "wheel>=0.43.0",
    "loguru>=0.7.2",
    "scikit-image>=0.21.0",
    "scipy>=1.11.4",
    "segment-anything>=1.0",
    "piexif>=1.1.3",
    "GitPython>=3.1.43",
    "qtpy>=2.4.1",
    "pyqt6>=6.5.0",
    "pyqt6-qt6>=6.5.0",
    "pyqtgraph>=0.13.7",
    "pytest>=8.2.0",
    "ruff>=0.4.4",
    "pylint>=3.2.1",
    "syrupy>=4.6.1",
    "pytest-cov>=5.0.0",
    "coverage>=7.5.2",
    "contexttimer>=0.3.3",
    "pydub>=0.23.0",
]

pattern = re.compile(r"^([^@!=<>~;\[]+(?:\[[^\]]+\])?)(?:[@!=<>~;].*)?$")

deps = {match[0]: x for x in _deps for match in [pattern.findall(x)] if match}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


class DepsTableUpdateCommand(Command):
    """
    A custom distutils command that updates the dependency table.
    usage: python setup.py deps_table_update
    """

    description = "build runtime dependency table"
    user_options = [
        ("dep-table-update", None, "updates src/deforum/dependency_versions_table.py"),
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        entries = "\n".join([f'    "{k}": "{v}",' for k, v in deps.items()])
        content = [
            "# THIS FILE HAS BEEN AUTOGENERATED. To update:",
            "# 1. modify the `_deps` dict in setup.py",
            "# 2. run `make deps_table_update`",
            "deps = {",
            entries,
            "}",
            "",
        ]
        target = "src/deforum/dependency_versions_table.py"
        print(f"updating {target}")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(content))


extras = {}

install_requires = deps_list(
    "torch",
    "torchvision",
    "einops",
    "numexpr",
    "matplotlib",
    "pandas",
    "av",
    "pims",
    "imageio-ffmpeg",
    "rich",
    "gdown",
    "py3d",
    "librosa",
    "numpy",
    "opencv-python-headless",
    "timm",
    "transformers",
    "omegaconf",
    "aiohttp",
    "scipy",
    "psutil",
    "clip-interrogator",
    "streamlit",
    "torchsde",
    "fastapi",
    "diffusers",
    "accelerate",
    "python-decouple",
    "imageio[ffmpeg]",
    "xformers",
    "kornia",
    "tensorrt",
    "onnx_graphsurgeon",
    "zstandard",
    "onnx",
    "polygraphy",
    "wheel",
    "loguru",
    "mutagen",
    "scikit-image",
    "segment-anything",
    "piexif",
    "GitPython",
    "contexttimer",
    "pydub",
)

extras["dev"] = deps_list("pytest", "ruff", "pylint", "syrupy", "pytest-cov", "coverage")

extras["comfy"] = deps_list(
    "einops",
    "numexpr",
    "matplotlib",
    "pandas",
    "av",
    "pims",
    "imageio-ffmpeg",
    "rich",
    "gdown",
    "py3d",
    "librosa",
)

setup(
    name="deforum",
    version="0.1.9.dev1",
    description="State-of-the-art Animation Diffusion in PyTorch and TRT.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning diffusion deforum pytorch stable diffusion",
    license="Apache",
    author="The Deforum team",
    author_email="deforum-art@deforum.com",
    url="https://github.com/deforum-studio/deforum",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.10,<3.15",
    install_requires=list(install_requires),
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "deforum=deforum.commands.deforum_cli:start_deforum_cli",
            "deforum-test=deforum.commands.deforum_test:start_deforum_test",
            "deforum-profile=deforum.commands.deforum_profiling:start_deforum_test",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    cmdclass={"deps_table_update": DepsTableUpdateCommand},
)