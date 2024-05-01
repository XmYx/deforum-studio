import os
import platform
from dataclasses import dataclass
from decouple import config

def get_os():
    return {"Windows": "Windows", "Linux": "Linux", "Darwin": "Mac"}.get(platform.system(), "Unknown")

# Typesafe config data loaded from .env or .ini
@dataclass
class LogConfig:
    log_level: str
    log_to_file: bool
    log_file: str
    log_max_bytes: int
    log_backup_count: int

@dataclass
class AppConfig(LogConfig):
    root_path: str
    src_path: str
    comfy_path: str
    settings_path: str
    model_dir: str
    other_model_dir: str
    output_dir: str
    comfy_update: bool
    allow_blocking_input_frame_lists: bool
    projectm_docker_image: str

    @staticmethod
    def load():
        # The path under which all Deforum non-code assets will be stored (logs, outputs, models etc...)
        # defaults to <HOME>/deforum. Other paths configured below depend on this value unless overriden.
        default_root_path = os.path.join(os.getenv('HOME'), 'deforum')
        root_path = config('ROOT_PATH', default_root_path)
        os.makedirs(root_path, exist_ok=True)
        
        # Determine the top-level location of this codebase. By default, ComfyUI is expected to be checked out within this directory.
        # TODO: would be more logical to expect ComfyUI to be checked out in the same directory as Deforum rather than within the deforum directory.
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        deforum_dir = os.path.dirname(utils_dir)
        default_src_path = os.path.dirname(deforum_dir)
        src_path = config('SRC_PATH', default_src_path)

        return AppConfig(
            root_path = root_path,
            src_path = src_path,
            comfy_path = config('COMFY_PATH', default=os.path.join(src_path, "ComfyUI")),
            settings_path = config('SETTINGS_PATH', default=os.path.join(root_path, "settings")),
            model_dir = config('MODEL_PATH', default=os.path.join(root_path, "models")),
            other_model_dir = config('OTHER_MODEL_PATH', default=os.path.join(root_path, "models/other")),
            output_dir = config('OUTPUT_PATH', default=os.path.join(root_path, "output/deforum")),
            comfy_update = config('COMFY_UPDATE', default=False, cast=bool),
            allow_blocking_input_frame_lists = config('ALLOW_BLOCKING_INPUT_FRAME_LISTS', default=False, cast=bool),
            projectm_docker_image = config('PROJECTM_DOCKER_IMAGE', default="rewbs/projectm-cli:0.0.4"),
            log_level = config('DEFORUM_LOG_LEVEL', default='DEBUG'),
            log_to_file = config('DEFORUM_LOG_TO_FILE', default=False, cast=bool),
            log_file = config('DEFORUM_LOG_FILE',  default=os.path.join(root_path,'logs/app.log')),
            log_max_bytes =  config('DEFORUM_LOG_FILE', cast=int, default=10485760),
            log_backup_count = config('DEFORUM_LOG_BACKUP_COUNT', cast=int, default=10485760)
        )

# Make available for import 
config = AppConfig.load()