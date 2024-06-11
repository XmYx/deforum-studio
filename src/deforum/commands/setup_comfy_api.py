import argparse
import os
import subprocess
import yaml
import requests

# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Automatically set up ComfyUI and its components based on a YAML configuration.')
parser.add_argument('config_path', type=str, help='Path to the YAML configuration file.')
parser.add_argument('comfy_path', type=str, help='Path to the ComfyUI directory.')
args = parser.parse_args()

# Load the YAML configuration
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

# Function to run shell commands
def run_cmd(command):
    subprocess.run(command, shell=True, check=False)

# Function to download files with checks
def download_file(url, dest_path):
    if not os.path.isfile(dest_path):
        response = requests.get(url, allow_redirects=True)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as file:
            file.write(response.content)

# Create additional directories if provided
if 'additional_dirs' in config:
    for dir in config["additional_dirs"]:
        os.makedirs(os.path.join(args.comfy_path, dir), exist_ok=True)

# Install global requirements if provided
if 'requirements' in config:
    for requirement in config["requirements"]:
        run_cmd(f"pip install {requirement}")

# Clone custom nodes if provided
custom_nodes_dir = os.path.join(args.comfy_path, "custom_nodes")
os.makedirs(custom_nodes_dir, exist_ok=True)
if 'custom_nodes' in config:
    for node in config['custom_nodes']:
        node_repo_url = f"https://github.com/{node}"
        run_cmd(f"git clone {node_repo_url} {os.path.join(custom_nodes_dir, node.split('/')[1])}")

# Install requirements for ComfyUI and all custom_nodes
directories_to_check = [args.comfy_path] + [os.path.join(custom_nodes_dir, node.split('/')[1]) for node in config.get('custom_nodes', [])]
for dir_path in directories_to_check:
    requirements_path = os.path.join(dir_path, 'requirements.txt')
    if os.path.exists(requirements_path):
        run_cmd(f"pip install -r {requirements_path}")

# Download CivitAI models if provided
if 'civitai' in config and 'models' in config['civitai']:
    for model in config['civitai']['models']:
        civitai_url = f"https://civitai.com/api/download/models/{model['id']}?token={config['civitai']['token']}"
        download_file(civitai_url, os.path.join(args.comfy_path, "models", model['dest']))

# Download HuggingFace models if provided
if 'huggingface' in config and 'models' in config['huggingface']:
    for model in config['huggingface']['models']:
        hf_url = f"https://huggingface.co/{model['id']}/resolve/main/{model['path']}"
        download_file(hf_url, os.path.join(args.comfy_path, "models", model['dest']))

print("Setup completed successfully.")
