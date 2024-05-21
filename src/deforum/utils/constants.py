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
    log_dir: str
    log_max_bytes: int
    log_backup_count: int

@dataclass
class AppConfig(LogConfig):
    root_path: str
    src_path: str
    comfy_path: str
    settings_path: str
    presets_path: str
    model_dir: str
    other_model_dir: str
    output_dir: str
    video_dir: str
    comfy_update: bool
    enable_onediff: bool
    allow_blocking_input_frame_lists: bool
    projectm_executable: str

    @staticmethod
    def load():
        # The path under which all Deforum non-code assets will be stored (logs, outputs, models etc...)
        # defaults to <HOME>/deforum. Other paths configured below depend on this value unless overriden.
        default_root_path = os.path.join(os.path.expanduser('~'), 'deforum')
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
            presets_path =  config('PRESETS_PATH', default=os.path.join(root_path, "presets")),
            model_dir = config('MODEL_PATH', default=os.path.join(root_path, "models")),
            other_model_dir = config('OTHER_MODEL_PATH', default=os.path.join(root_path, "models", "other")),
            output_dir = config('OUTPUT_PATH', default=os.path.join(root_path, "output", "deforum")),
            video_dir = config('VIDEO_PATH', default=os.path.join(root_path, "output", "video")),
            comfy_update = config('COMFY_UPDATE', default=False, cast=bool),
            enable_onediff = config('ENABLE_ONEDIFF', default=True, cast=bool),
            allow_blocking_input_frame_lists = config('ALLOW_BLOCKING_INPUT_FRAME_LISTS', default=False, cast=bool),
            projectm_executable = config('PROJECTM_EXECUTABLE', default="projectMCli"),
            log_level = config('DEFORUM_LOG_LEVEL', default='DEBUG'),
            log_to_file = config('DEFORUM_LOG_TO_FILE', default=False, cast=bool),
            log_dir = config('DEFORUM_LOG_DIR',  default=os.path.join(root_path,'logs')),
            log_max_bytes =  config('DEFORUM_LOG_FILE', cast=int, default=10485760),
            log_backup_count = config('DEFORUM_LOG_BACKUP_COUNT', cast=int, default=10485760)
        )

# Make available for import 
config = AppConfig.load()
# root_path = config.root_path
