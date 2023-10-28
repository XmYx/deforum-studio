import platform

root_path = ""

def get_os():
    return {"Windows": "Windows", "Linux": "Linux", "Darwin": "Mac"}.get(platform.system(), "Unknown")

import os

# root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
comfy_path = os.path.join(root_path, "src/ComfyUI")

model_dir = os.path.join(root_path, "models/checkpoints")
other_model_dir = os.path.join(root_path, "models/other")
