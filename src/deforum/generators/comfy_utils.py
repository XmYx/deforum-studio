import os
import subprocess
import sys

from deforum.utils.constants import comfy_path


def ensure_comfy():
    # 1. Check if the "src" directory exists
    # if not os.path.exists(os.path.join(root_path, "src")):
    #     os.makedirs(os.path.join(root_path, 'src'))
    # 2. Check if "ComfyUI" exists
    if not os.path.exists(comfy_path):
        # Clone the repository if it doesn't exist
        subprocess.run(["git", "clone", "https://github.com/comfyanonymous/ComfyUI", comfy_path])
    else:
        current_folder = os.getcwd()
        os.chdir(comfy_path)
        subprocess.run(["git", "pull"])

        os.chdir(current_folder)

    sys.path.append(comfy_path)
