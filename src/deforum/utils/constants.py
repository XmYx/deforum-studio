import os
import platform
from dataclasses import dataclass
from decouple import config

# Retrieve the home directory using the HOME environment variable
home_dir = os.getenv('HOME')

# Define the path for the 'deforum' directory within the home directory
root_path = os.path.join(home_dir, 'deforum')

# Check if the directory exists, and create it if it does not
if not os.path.exists(root_path):
    os.makedirs(root_path)


utils_dir = os.path.dirname(os.path.abspath(__file__))
deforum_dir = os.path.dirname(utils_dir)
src_dir = os.path.dirname(deforum_dir)

def get_os():
    return {"Windows": "Windows", "Linux": "Linux", "Darwin": "Mac"}.get(platform.system(), "Unknown")


# Typesafe config data loaded from .env or .ini
@dataclass
class AppConfig:
    comfy_path: str
    model_dir: str
    other_model_dir: str
    output_dir: str
    comfy_update: bool
    allow_blocking_input_frame_lists: bool
    projectm_docker_image: str

    @staticmethod
    def load():
        return AppConfig(
            comfy_path = config('COMFY_PATH', default=os.path.join(src_dir, "ComfyUI")),
            model_dir = config('MODEL_PATH', default=os.path.join(root_path, "models")),
            other_model_dir = config('OTHER_MODEL_PATH', default=os.path.join(root_path, "models/other")),
            output_dir = config('OUTPUT_PATH', default=os.path.join(root_path, "output/deforum")),
            comfy_update = config('COMFY_UPDATE', default=False, cast=bool),
            allow_blocking_input_frame_lists = config('ALLOW_BLOCKING_INPUT_FRAME_LISTS', default=False, cast=bool),
            projectm_docker_image = config('PROJECTM_DOCKER_IMAGE', default="rewbs/projectm-cli:0.0.4"),
        )

# Make available for import 
config = AppConfig.load()
