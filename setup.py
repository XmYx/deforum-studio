import platform
import re
import sys
from distutils.core import Command
from setuptools import find_packages, setup

python_version = '.'.join(map(str, sys.version_info[:2]))
os_name = platform.system().lower()

torch_package_urls = {
    '3.10': {
        'linux': 'torch-2.3.0%2Bcu121-cp310-cp310-linux_x86_64.whl',
        'windows': 'torch-2.3.0%2Bcu121-cp310-cp310-win_amd64.whl'
    },
    '3.11': {
        'linux': 'torch-2.3.0%2Bcu121-cp311-cp311-linux_x86_64.whl',
        'windows': 'torch-2.3.0%2Bcu121-cp311-cp311-win_amd64.whl'
    },
    '3.8': {
        'linux': 'torch-2.3.0%2Bcu121-cp38-cp38-linux_x86_64.whl',
        'windows': 'torch-2.3.0%2Bcu121-cp38-cp38-win_amd64.whl'
    },
    '3.9': {
        'linux': 'torch-2.3.0%2Bcu121-cp39-cp39-linux_x86_64.whl',
        'windows': 'torch-2.3.0%2Bcu121-cp39-cp39-win_amd64.whl'
    }
}

torchvision_package_urls = {
    '3.10': {
        'linux': 'torchvision-0.18.0%2Bcu121-cp310-cp310-linux_x86_64.whl',
        'windows': 'torchvision-0.18.0%2Bcu121-cp310-cp310-win_amd64.whl'
    },
    '3.11': {
        'linux': 'torchvision-0.18.0%2Bcu121-cp311-cp311-linux_x86_64.whl',
        'windows': 'torchvision-0.18.0%2Bcu121-cp311-cp311-win_amd64.whl'
    },
    '3.8': {
        'linux': 'torchvision-0.18.0%2Bcu121-cp38-cp38-linux_x86_64.whl',
        'windows': 'torchvision-0.18.0%2Bcu121-cp38-cp38-win_amd64.whl'
    },
    '3.9': {
        'linux': 'torchvision-0.18.0%2Bcu121-cp39-cp39-linux_x86_64.whl',
        'windows': 'torchvision-0.18.0%2Bcu121-cp39-cp39-win_amd64.whl'
    }
}

if python_version in torch_package_urls:
    torch_url = torch_package_urls[python_version][os_name]
    torchvision_url = torchvision_package_urls[python_version][os_name]
else:
    sys.exit(f"Unsupported Python version: {python_version}")

torch_path = f"https://download.pytorch.org/whl/cu121/{torch_url}"
torchvision_path = f"https://download.pytorch.org/whl/cu121/{torchvision_url}"

# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/deforum/dependency_versions_table.py
_deps = [
    f'torch@{torch_path}',
    f'torchvision@{torchvision_path}',
    'einops==0.6.0',
    'numexpr==2.8.4',
    'matplotlib==3.7.1',
    'pandas==1.5.3',
    'av==10.0.0',
    'pims==0.6.1',
    'imageio-ffmpeg==0.4.8',
    'rich==13.3.2',
    'gdown==4.7.1',
    'py3d==0.0.87',
    'librosa==0.10.0.post2',
    'numpy==1.26.4',
    'opencv-python-headless',
    'timm==0.6.13',
    'transformers==4.40.2',
    'omegaconf==2.3.0',
    'aiohttp==3.9.3',
    'psutil==5.9.6',
    'clip-interrogator==0.6.0',
    'streamlit==1.27.2',
    'torchsde>=0.2.5',
    'fastapi>=0.100.0',
    'diffusers==0.27.2',
    'accelerate==0.29.3',
    'python-decouple>=3.8',
    'mutagen>=1.47.0',
    'imageio[ffmpeg]==2.34.1',
    'xformers==0.0.26.post1',
    'tensorrt==10.0.1',
    'onnx_graphsurgeon==0.5.2',
    'onnx==1.16.0',
    'zstandard==0.22.0',
    'polygraphy==0.49.9',
    'kornia==0.7.2',
    'wheel==0.43.0',
    'loguru==0.7.2',
    'scikit-image==0.21.0',
    'scipy==1.11.4',
    'segment-anything==1.0',
    'piexif==1.1.3',
    'GitPython==3.1.43',
    'qtpy==2.4.1',
    'pyqt6==6.5.0',
    'pyqt6-qt6==6.5.0',
    'pyqtgraph==0.13.7',
    'pytest>=8.2.0',
    'ruff>=0.4.4',
    'pylint>=3.2.1',
    'syrupy>=4.6.1',
    'pytest-cov>=5.0.0',
    'coverage>=7.5.2',
    'librosa>=0.10.0.post2',
    'contexttimer>=0.3.3',
    'pydub>=0.23.0'
]

# this is a lookup table with items like:
#
# tokenizers: "huggingface-hub==0.8.0"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
pattern = re.compile(r"^([^@!=<>~]+)(?:[@!=<>~].*)?$")

deps = {match[0]: x for x in _deps for match in [pattern.findall(x)] if match}


# since we save this data in src/deforum/dependency_versions_table.py it can be easily accessed from
# anywhere. If you need to quickly access the data from this table in a shell, you can do so easily with:
#
# python -c 'import sys; from deforum.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets
#
# Just pass the desired package names to that script as it's shown with 2 packages above.
#
# If deforum is not yet installed and the work is done from the cloned repo remember to add `PYTHONPATH=src` to the
# script above
#
# You can then feed this for example to `pip`:
#
# pip install -U $(python -c 'import sys; from deforum.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets)
#

def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


class DepsTableUpdateCommand(Command):
    """
    A custom distutils command that updates the dependency table.
    usage: python setup.py deps_table_update
    """

    description = "build runtime dependency table"
    user_options = [
        # format: (long option, short option, description).
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
            "# 2. run `make deps_table_update``",
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

install_requires = deps_list('torch',
                             'torchvision',
                             'einops',
                             'numexpr',
                             'matplotlib',
                             'pandas',
                             'av',
                             'pims',
                             'imageio-ffmpeg',
                             'rich',
                             'gdown',
                             'py3d',
                             'librosa',
                             'numpy',
                             'opencv-python-headless',
                             'timm',
                             'transformers',
                             'omegaconf',
                             'aiohttp',
                             'scipy',
                             'psutil',
                             'clip-interrogator',
                             'streamlit',
                             'torchsde',
                             'fastapi',
                             'diffusers',
                             'accelerate',
                             'python-decouple',
                             'imageio[ffmpeg]',
                             'xformers',
                             'kornia',
                             'tensorrt',
                             'onnx_graphsurgeon',
                             'zstandard',
                             'onnx',
                             'polygraphy',
                             'wheel',
                             'loguru',
                             'mutagen',
                             'scikit-image',
                             'segment-anything',
                             'piexif',
                             'GitPython',
                             'qtpy',
                             'pyqt6',
                             'pyqt6-qt6',
                             'pyqtgraph',
                             'librosa',
                             'contexttimer',
                             'pydub'
                             )

extras['dev'] = deps_list('pytest', 'ruff', 'pylint', 'syrupy', 'pytest-cov', 'coverage')


setup(
    name="deforum",
    version="0.01.8.dev1",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
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
    # package_data={"deforum": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=list(install_requires),
    extras_require=extras,
    entry_points={"console_scripts": ["deforum=deforum.commands.deforum_cli:start_deforum_cli",
                                      "deforum-test=deforum.commands.deforum_test:start_deforum_test",
                                      "deforum-profile=deforum.commands.deforum_profiling:start_deforum_test"
                                      ]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    cmdclass={"deps_table_update": DepsTableUpdateCommand},
)

# Release checklist
# 1. Change the version in __init__.py and setup.py.
# 2. Commit these changes with the message: "Release: Release"
# 3. Add a tag in git to mark the release: "git tag RELEASE -m 'Adds tag RELEASE for pypi' "
#    Push the tag to git: git push --tags origin main
# 4. Run the following commands in the top-level directory:
#      python setup.py bdist_wheel
#      python setup.py sdist
# 5. Upload the package to the pypi test server first:
#      twine upload dist/* -r pypitest
#      twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
# 6. Check that you can install it in a virtualenv by running:
#      pip install -i https://testpypi.python.org/pypi deforum
#      deforum env
#      deforum test
# 7. Upload the final version to actual pypi:
#      twine upload dist/* -r pypi
# 8. Add release notes to the tag in GitHub once everything is looking deforumish.
# 9. Update the version in __init__.py, setup.py to the new version "-dev" and push to master
