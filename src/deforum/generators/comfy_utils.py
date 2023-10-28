import os
import subprocess
import sys

from deforum.utils.constants import comfy_path, root_path


def ensure_comfy():
    # 1. Check if the "src" directory exists
    # if not os.path.exists(os.path.join(root_path, "src")):
    #     os.makedirs(os.path.join(root_path, 'src'))
    # 2. Check if "ComfyUI" exists
    if not os.path.exists(comfy_path):
        # Clone the repository if it doesn't exist
        subprocess.run(["git", "clone", "https://github.com/comfyanonymous/ComfyUI", comfy_path])
    else:
    #     # 3. If "ComfyUI" does exist, check its commit hash
        current_folder = os.getcwd()
        os.chdir(comfy_path)
        subprocess.run(["git", "pull"])

    #     current_commit = subprocess.getoutput("git rev-parse HEAD")
    #
    #     # 4. Reset to the desired commit if necessary
    #     if current_commit != "4185324":  # replace with the full commit hash if needed
    #         subprocess.run(["git", "fetch", "origin"])
    #         subprocess.run(["git", "reset", "--hard", "b935bea3a0201221eca7b0337bc60a329871300a"])  # replace with the full commit hash if needed
    #         subprocess.run(["git", "pull", "origin", "master"])
        os.chdir(current_folder)

    sys.path.append(comfy_path)