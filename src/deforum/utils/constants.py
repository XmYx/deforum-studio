import os
import platform

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


# root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
comfy_path = os.path.join(src_dir, "ComfyUI")

model_dir = os.path.join(root_path, "models/checkpoints")
other_model_dir = os.path.join(root_path, "models/other")
