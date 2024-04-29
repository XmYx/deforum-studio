# Deforum
State-of-the-art Animation Diffusion in PyTorch and TRT.
## Installation
You can install Deforum using one of the following methods:
### Virtual Environment
Create a virtual environement with venv or conda
```bash
python -m venv venv
source venv/bin/activate
```
with conda:
```bash
conda create -n deforum python=3.10
conda actiave deforum
```
### PyPI
Install from PyPI using `pip`:
```bash
pip install deforum
```
### From Source
Install from github with the following commands:
```bash
git clone https://github.com/deforum-studio/deforum
cd deforum
pip install -e .["cli"]
```
### For Developers
Install with `["dev"]` for developer dependencies:
```bash
git clone https://github.com/deforum-studio/deforum
cd deforum
pip install -e .["dev"]
```

## Testing
Test install by running the test animation pipeline
```bash
COMFY_PATH=src/ComfyUI python tests/test_animation_pipeline.py
```
Test diffusers pipeline
```bash
python tests/test_diffusers_pipeline.py
```

## WebUI
Launch webui with the following command
```bash
deforum webui
```

### Instructions for WebUI
Upload the deforum.txt file into the WebUI from the Presets folder.
Now, you will need to make sure to check `Use Settings File`.
This is because the settings filed gets loaded into the UI that allows you to edit the parameters but you can also use the settings file straight.

## License
Deforum is licensed under the GNU General Public License v3.0 License.

For more information please refer to [license](https://github.com/deforum-studio/deforum/blob/main/LICENSE).
