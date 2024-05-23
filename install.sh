#!/bin/bash

# ASCII Art
echo "
[installer for linux version 0.1.0]

██████╗ ███████╗███████╗ ██████╗ ██████╗ ██╗   ██╗███╗   ███╗
██╔══██╗██╔════╝██╔════╝██╔═══██╗██╔══██╗██║   ██║████╗ ████║
██║  ██║█████╗  █████╗  ██║   ██║██████╔╝██║   ██║██╔████╔██║
██║  ██║██╔══╝  ██╔══╝  ██║   ██║██╔══██╗██║   ██║██║╚██╔╝██║ 
██████╔╝███████╗██║     ╚██████╔╝██║  ██║╚██████╔╝██║ ╚═╝ ██║
╚═════╝ ╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝

Welcome to the Deforum Animation Toolkit!

Deforum is a powerful tool for generating videos using AI. 
It leverages Diffusion checkpoints and various functions
to create dynamic, evolving visuals.

==============
INSTALL CONFIG
==============
"

# Function to create directory if it does not exist
create_dir_if_not_exists() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Default values
VENV_DIR="venv"
ROOT_DIR="$HOME/deforum"
COMFY_DIR="src/ComfyUI"
SILENT=0

# Parse arguments
while getopts ":s" opt; do
  case ${opt} in
    s )
      SILENT=1
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
  esac
done

# Collect VENV_PATH
if [ $SILENT -eq 0 ]; then
    read -p "Virtual environment path or ENTER to skip (default: $VENV_DIR): " VENV_INPUT
    VENV_DIR=${VENV_INPUT:-$VENV_DIR}
fi
create_dir_if_not_exists "$VENV_DIR"
VENV_PATH=$(realpath "$VENV_DIR")

# Collect ROOT_PATH
if [ $SILENT -eq 0 ]; then
    read -p "Root directory path or ENTER to skip (default: $ROOT_DIR): " ROOT_INPUT
    ROOT_DIR=${ROOT_INPUT:-$ROOT_DIR}
fi
create_dir_if_not_exists "$ROOT_DIR"
ROOT_PATH=$(realpath "$ROOT_DIR")

# Collect COMFY_PATH
if [ $SILENT -eq 0 ]; then
    read -p "ComfyUI directory path or ENTER to skip (default: $COMFY_DIR): " COMFY_INPUT
    COMFY_DIR=${COMFY_INPUT:-$COMFY_DIR}
fi
create_dir_if_not_exists "$COMFY_DIR"
COMFY_PATH=$(realpath "$COMFY_DIR")

# Save to .env file
echo "VENV_PATH=$VENV_PATH" > .env
echo "ROOT_PATH=$ROOT_PATH" >> .env
echo "COMFY_PATH=$COMFY_PATH" >> .env

git submodule update --init --recursive

# Create virtual environment and install dependencies
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
pip install -e .

echo "Installation completed successfully."
