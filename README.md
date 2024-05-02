# Deforum
State-of-the-art Animation Diffusion in PyTorch and TRT.

## System Requirements
- Linux-like operating system
- NVIDIA GPU with CUDA drivers

## Installation
Currently, the repository is in development, and the recommended installation method is "for developers."

### For Developers
1. Clone the repository:
   ```bash
   git clone https://github.com/deforum-studio/deforum
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
   pip install -e .["dev"]
   ```

## Testing
To ensure that the library is properly installed, run a test generation:
```bash
python tests/test_animation_pipeline.py
```
The generated outputs will be saved to `$HOME/deforum/outputs`, which is the default output path.

## CLI Commands
Deforum has the following CLI modes:
- `deforum ui`: PyQt6 UI for configuring and running animations
- `deforum webui`: Streamlit web UI for configuring and running animations
- `deforum animatediff`: Command-line tool for running animations
- `deforum runpresets --options randomize_files=True`: Run through all motion presets in `presets/` for testing purposes.
- `deforum api`: FastAPI server
- `deforum setup`: Install Stable-Fast optimizations

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