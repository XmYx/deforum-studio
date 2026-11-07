# Deforum
State-of-the-art Animation Diffusion in PyTorch, for every model from SD 1.5 to FLUX.3.

## System Requirements
- Python 3.11, 3.12, 3.13 or 3.14
- PyTorch 2.6+ (tested with 2.15) on any of:
  - NVIDIA GPU (CUDA 12.8 / 13.x wheels)
  - AMD GPU on Linux (ROCm wheels)
  - Apple Silicon (MPS)
  - CPU (slow, but works)
- Linux, Windows or macOS

## Supported models
Generation runs on ComfyUI's model code (bundled as a git submodule and used as a library), so every
image model family ComfyUI loads works in every Deforum mode (2D, 3D, Video Input, Interpolation):

| Family | What to put where |
|---|---|
| SD 1.x / 2.x, SDXL (and fine-tunes), SD3 all-in-one | single file in `models/checkpoints` |
| FLUX.1 dev/schnell, Chroma | diffusion model in `models/diffusion_models`, `clip_l` + `t5xxl` in `models/text_encoders`, `ae` VAE in `models/vae` |
| FLUX.2 dev / Klein | diffusion model, Mistral-3 (dev) or Qwen3 (Klein) text encoder, `flux2-vae` |
| FLUX.3 Image Klein | diffusion model, `qwen3vl_4b` text encoder, `flux2-vae` |
| Qwen-Image, Lumina 2 / Z-Image | diffusion model + its text encoder + VAE |

For split releases the matching text encoder(s) and VAE are found automatically by file name; they can
also be set explicitly (`text_encoders`, `vae`, `clip_type` settings). Guidance-distilled models
(FLUX.1-dev, FLUX.2-dev) receive the CFG scale schedule as distilled guidance. Flow-matching models get a
fitting schedule automatically when a Karras/exponential schedule is selected (`flux2` = the official
resolution-aware FLUX.2 schedule). FLUX.3 uses BFL's reference settings: euler / `simple`, CFG 1, flow shift ln 16.

Model folders: `~/deforum/models/<kind>`, the bundled `src/ComfyUI/models/<kind>`, and any existing
ComfyUI `models` folder listed in `EXTRA_MODEL_PATHS` (in `.env` or the environment).

## World Model mode (FLUX.3 Action)
`animation_mode = "World Model"` lets [FLUX.3 Action](https://huggingface.co/collections/black-forest-labs/flux-3-action)
predict the animation: from the current frame and the frame's prompt it predicts the next ~2 s of video
(the frames it normally discards when used as a robot policy). The input can be an init image, an init video
(optionally re-anchored on the video every chunk) or the animation's own frames, and each predicted frame can be
refined with the selected image model (`world_model_refine_denoise`, 0 = raw predictions).

It needs its own environment: run `bash install_worldmodel.sh` after accepting the (gated) model license and
`hf auth login` (it pins flux-action's validated torch 2.10 + CUDA 12.8 with the matching NATTEN kernels).
The DiT and video VAE take ~10 GB of VRAM; the 8.9 GB text encoder stays in RAM. FLUX.3 Action is trained on robot, driving and game footage, so predictions from artistic
frames drift toward that domain; image-model refinement keeps the look. Camera keyframes do not steer it
(that needs a FLUX.3 Action checkpoint with a camera/gaming action head).

## Installation

### For Users
- Linux / macOS: `bash install.sh`, then `bash start.sh`
- Windows: `install.bat`, then `start.bat`

The installer detects the hardware and installs the matching PyTorch build (CUDA, ROCm, MPS or CPU).
Override with `DEFORUM_TORCH_BACKEND` (`cu132`, `cu130`, `cu128`, `rocm7.14`, `cpu`, `mps`) and pick the
Python version with `DEFORUM_PYTHON` (default 3.12).

### For Developers
```bash
git clone --recursive https://github.com/deforum-studio/deforum
cd deforum
uv venv -p 3.12 && source .venv/bin/activate        # or python3.12 -m venv .venv
# 1. PyTorch for your hardware, e.g. CUDA 13.2 / ROCm / Apple (plain `pip install torch torchvision`)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
# 2. ComfyUI's requirements, then Deforum
uv pip install -r src/ComfyUI/requirements.txt
uv pip install -e ".[dev]"
```
Set `DEFORUM_DEVICE` (e.g. `cpu`, `cuda:1`, `mps`) to force a device.

### Performance notes
- All prompt embeddings of an animation are encoded before the first frame; the text encoder then leaves
  VRAM, so frames never wait on a text-encoder / diffusion-model swap (matters most for FLUX-class encoders).
- Models stay in RAM between renders; only VRAM is released after a run.
- Depth: Depth-Anything-V2 and MiDaS take ~10 ms/frame on an RTX 4090 at 1024x576 (LeReS ~20 ms, AdaBins ~50 ms,
  Zoe ~115 ms). Depth maps computed for saving are reused for the next frame's 3D warp.

## Configuration

See `src/deforum/utils/constants.py` for a list of configuration options. These can be set as environment variales, or defined in a `.env` or `settings.ini` file in the pwd of the python process. 

Configuraton is loaded with [python-decouple](https://pypi.org/project/python-decouple/) – see that documentation for details like [precedence ordering between env vars and config files](https://pypi.org/project/python-decouple/#how-does-it-work).

Configuration options include:

- `EXTRA_MODEL_PATHS`: one or more existing ComfyUI `models` folders (separated by `:` on Linux/macOS, `;` on Windows) to load models from.
- `DEFORUM_DEVICE`: force a device (`cpu`, `cuda:1`, `mps`, ...). Default: best available (CUDA/ROCm, MPS, XPU, CPU).
- `DEFORUM_FP16_ACCUMULATION`: fp16 matmul accumulation on NVIDIA, ~15-20% faster sampling. Default: on; set `0` to disable.
- `ROOT_PATH` (aka deforum storage path): root path under which deforum will store and search all non-code assets like models, settings files, output videos etc.... Default: `~/deforum`
- `PRESETS_PATH`: directory in which tests and run commands expect to find presets as defined by `https://github.com/deforum-studio/deforum-presets.git`. Default: `{ROOT_DIR}/presets`
- `SETTINGS_PATH`: output directory for settings files. Default: `{ROOT_DIR}/settings`
- `OUTPUT_PATH`: output directory for generation results (images, intermediaries etc...). Default: `{ROOT_DIR}/output`
- `VIDEO_PATH`: output directory for videos. Default:  `{ROOT_DIR}/output/videos`
- `DEFORUM_LOG_LEVEL`: log level. Default: `DEBUG`.

This is a **non-exhaustive** list. See `src/deforum/utils/constants.py` for more.


## Unit testing

These tests run in CI on every commit on free hardware, so NEVER add unit tests that require a GPU. Mock away!

### Setup

Run the following to install test dependencies:

```bash
pip install -e .['dev']
```

### Execution

Unit tests are under `./tests/unittests`. Point your IDE's test mechanism to this directory or the parent to include `integrationtests` as well. Alternatively, you can run them from the command line with:

```bash
pytest tests/unittests
```

## Integration testing

These tests should be fast enough to run frequently as part of your dev loop, but require a GPU so don't yet run in CI.

### Setup

Run the following to install test dependencies:

```bash
pip install -e .['dev']
```

### Execution

Integration tests are under `./tests/integrationtests`. Point your IDE's test mechanism to this directory or the parent to include `unittests` as well. Alternatively, you can run them from the command line with:

```bash
pytest tests/integrationtests
```


## End-to-end testing

There are full tests that require a GPU and can take hours to run depending on the settings files involved.

### Setup

Prior to testing make sure you clone the `https://github.com/deforum-studio/deforum-presets.git` into `PRESETS_PATH`, which defaults to `~/deforum/presets`. Therefore, by default, the settings files within that repo assume a path of `~/deforum/presets/settings/`.

You can verify that the paths are setup correctly by running `python tests/test_animation_pipeline.py`. 

### Execution

- Run `deforum run-all` for a full run of all settings files under `{PRESETS_PATH}/settings`.
- Run `deforum test-e2e` for a full run of all settings files under `{PRESETS_PATH}/settings`, and compare results to a baseline (or create a new baseline if none exists).

## Ad-hoc testing

For ad-hoc validation of miscellaneous functionality, you can directly run the test-* scripts under `./examples`.
(TODO: these should probably converted to integration tests with clear expected outputs. Alternatively we can recast these scripts as "examples" of how to use the lib.)


## Linting 

We use `ruff` for formatting & basic linting, and `pylint` for a small number of checks that are not supported by ruff. See `pyproject.toml` and `.pylintrc` for their respective config.
When working on the code, please set up your IDE accordingly to avoid build failures. 

We currently have many linting issues to fix, but are using a ratcheting mechanism in CI to ensure the number does not increase.


## CLI Commands
Deforum has the following CLI modes:
- `deforum ui`: PyQt6 UI for configuring and running animations
- `deforum webui`: Streamlit web UI for configuring and running animations
- `deforum animatediff`: Command-line tool for running animations
- `deforum api`: FastAPI server
- `deforum runsingle --file ~/deforum/presets/preset.txt`: Run single settings file
- `deforum config`
- `deforum test-e2e`: Run all settings files under `{PRESETS_PATH}/settings` (defaults to `~/deforum/presets/settings/`) 
- `deforum run-all`: Run all settings files under `{PRESETS_PATH}/settings` (defaults to `~/deforum/presets/settings/`) and compare to or create a baseline. 


## Documentation
Documentation for Deforum is currently a work in progress and will be included as part of this library in the future.

## Known Issues and Limitations
Please be aware that the codebase is changing rapidly, and it is currently difficult to assess if all features are working properly. However, to our knowledge, the code is functioning as intended.

## Contributing
Developers who wish to contribute to Deforum should create a branch from the `main` branch and label it with the format `username/branch-name`. All pull requests should be made into the `develop` branch.

## Acknowledgments and References
There are numerous references and acknowledgments that need to be made for the libraries and resources used in Deforum. This section is currently a work in progress and will be updated in the future.

## License
Deforum is licensed under the GNU General Public License v3.0 License. For more information, please refer to the [license](https://github.com/deforum-studio/deforum/blob/main/LICENSE).

## TODO Checklist
- [ ] Complete comprehensive documentation
- [ ] Add more test cases and examples
- [ ] Improve error handling and logging
- [ ] Optimize performance and resource usage
- [ ] Enhance user experience and usability
- [ ] Implement additional features and functionalities
- [ ] Refactor and clean up codebase
- [ ] Update acknowledgments and references
