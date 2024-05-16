import os
import time
from deforum import DeforumAnimationPipeline
from deforum.utils.constants import config
from deforum.pipeline_utils import load_settings

preset_dir = os.path.join(config.root_path,"presets")
settings_dir = os.path.join(preset_dir,"settings")
settings_file = "Zoom-To-New.txt"
settings_path = os.path.join(settings_dir,settings_file)

pipeline = DeforumAnimationPipeline.from_civitai("125703")

args = load_settings(settings_path)
args["video_init_path"] = os.path.join(settings_dir, args["video_init_path"])
print(args["video_init_path"])

start_time = time.time()
pipeline(**args)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")