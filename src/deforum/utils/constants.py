import os
import platform

root_path = ""

utils_dir = os.path.dirname(os.path.abspath(__file__))

deforum_dir = os.path.dirname(utils_dir)

src_dir = os.path.dirname(deforum_dir)


def get_os():
    return {"Windows": "Windows", "Linux": "Linux", "Darwin": "Mac"}.get(platform.system(), "Unknown")


# root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
comfy_path = os.path.join(src_dir, "ComfyUI")

model_dir = os.path.join(root_path, "models/checkpoints")
other_model_dir = os.path.join(root_path, "models/other")
