# Deforum
State-of-the-art Animation Diffusion in PyTorch and TRT.

## System Requirements
- Linux-like operating system
- NVIDIA GPU with CUDA drivers

## Installation
Currently, the repository is in development, but we already provide an easy installer for Windows and Linux.

### For Users:
On windows, run start.bat
On linux run bash start.sh

### For Developers
1. Clone the repository:
   ```bash
   git clone --recursive https://github.com/deforum-studio/deforum
   cd deforum
   ```
2. Create a virtual environment for `python==3.10`:
   - Using venv:
     ```bash
     virtualenv venv -p 3.10
     source venv/bin/activate
     ```
   - Using conda:
     ```bash
     conda create -n deforum python=3.10
     conda activate deforum
     ```
3. Install the library with developer dependencies:
   ```bash
   pip install -e .
   ```

## Configuration

See `src/deforum/utils/constants.py` for a list of configuration options. These can be set as environment variales, or defined in a `.env` or `settings.ini` file in the pwd of the python process. 

Configuraton is loaded with [python-decouple](https://pypi.org/project/python-decouple/) – see that documentation for details like [precedence ordering between env vars and config files](https://pypi.org/project/python-decouple/#how-does-it-work).

Configuration options include:

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
- `deforum setup`: Install Stable-Fast optimizations
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
