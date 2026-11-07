#!/usr/bin/env bash
# Deforum installer for Linux and macOS.
#
#   bash install.sh            # interactive
#   bash install.sh -s         # silent, defaults
#
# Environment overrides:
#   DEFORUM_PYTHON=3.12        Python version (3.11 - 3.14)
#   DEFORUM_TORCH_BACKEND=...  cu132 | cu130 | cu128 | rocm7.14 | cpu | mps   (default: auto-detect)
#   DEFORUM_TORCH_VERSION=...  torch version (default: latest)
set -euo pipefail

echo "
██████╗ ███████╗███████╗ ██████╗ ██████╗ ██╗   ██╗███╗   ███╗
██╔══██╗██╔════╝██╔════╝██╔═══██╗██╔══██╗██║   ██║████╗ ████║
██║  ██║█████╗  █████╗  ██║   ██║██████╔╝██║   ██║██╔████╔██║
██║  ██║██╔══╝  ██╔══╝  ██║   ██║██╔══██╗██║   ██║██║╚██╔╝██║
██████╔╝███████╗██║     ╚██████╔╝██║  ██║╚██████╔╝██║ ╚═╝ ██║
╚═════╝ ╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝
"

cd "$(dirname "$0")"

VENV_DIR="venv"
ROOT_DIR="$HOME/deforum"
COMFY_DIR="src/ComfyUI"
EXTRA_MODELS=""
PYVER="${DEFORUM_PYTHON:-3.12}"
SILENT=0
while getopts ":s" opt; do
  case ${opt} in
    s) SILENT=1 ;;
    *) echo "usage: $0 [-s]" >&2; exit 1 ;;
  esac
done

ask() {  # ask <prompt> <default> -> echoes the answer
  local answer=""
  if [ "$SILENT" -eq 0 ]; then read -r -p "$1 (default: $2): " answer; fi
  echo "${answer:-$2}"
}

VENV_DIR=$(ask "Virtual environment path" "$VENV_DIR")
ROOT_DIR=$(ask "Deforum data folder (models, outputs)" "$ROOT_DIR")
COMFY_DIR=$(ask "ComfyUI folder (bundled submodule)" "$COMFY_DIR")
EXTRA_MODELS=$(ask "Existing ComfyUI 'models' folder to reuse, or empty" "$EXTRA_MODELS")

mkdir -p "$ROOT_DIR/models/checkpoints" "$ROOT_DIR/models/diffusion_models" "$ROOT_DIR/models/text_encoders" \
         "$ROOT_DIR/models/vae" "$ROOT_DIR/models/loras"
ROOT_PATH=$(cd "$ROOT_DIR" && pwd)

# --- torch backend -------------------------------------------------------------------------
detect_backend() {
  if [ "$(uname -s)" = "Darwin" ]; then echo "mps"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then echo "cu132"; return; fi
  if command -v rocminfo >/dev/null 2>&1 || [ -e /dev/kfd ]; then echo "rocm7.14"; return; fi
  echo "cpu"
}
BACKEND="${DEFORUM_TORCH_BACKEND:-$(detect_backend)}"
case "$BACKEND" in
  mps)       TORCH_INDEX="" ;;   # the default PyPI wheels include Apple MPS support
  cpu)       TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
  cu*|rocm*) TORCH_INDEX="https://download.pytorch.org/whl/$BACKEND" ;;
  *) echo "Unknown DEFORUM_TORCH_BACKEND=$BACKEND" >&2; exit 1 ;;
esac
TORCH_SPEC="torch${DEFORUM_TORCH_VERSION:+==$DEFORUM_TORCH_VERSION}"
echo "Python $PYVER, torch backend: $BACKEND"

# --- ComfyUI -------------------------------------------------------------------------------
if [ "$COMFY_DIR" = "src/ComfyUI" ]; then
  git submodule update --init src/ComfyUI
fi
COMFY_PATH=$(cd "$COMFY_DIR" && pwd)

# --- environment ---------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  uv venv -p "$PYVER" "$VENV_DIR"
  PIP=(uv pip install -p "$VENV_DIR")
else
  "python$PYVER" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install -U pip
  PIP=("$VENV_DIR/bin/python" -m pip install)
fi
VENV_PATH=$(cd "$VENV_DIR" && pwd)

if [ -n "$TORCH_INDEX" ]; then
  "${PIP[@]}" "$TORCH_SPEC" torchvision --index-url "$TORCH_INDEX"
else
  "${PIP[@]}" "$TORCH_SPEC" torchvision
fi
"${PIP[@]}" -r "$COMFY_PATH/requirements.txt"
"${PIP[@]}" -e .

{
  echo "VENV_PATH=$VENV_PATH"
  echo "ROOT_PATH=$ROOT_PATH"
  echo "COMFY_PATH=$COMFY_PATH"
  if [ -n "$EXTRA_MODELS" ]; then echo "EXTRA_MODEL_PATHS=$EXTRA_MODELS"; fi
} > .env

"$VENV_PATH/bin/python" -c "import torch; from deforum.utils.device import get_device; print('torch', torch.__version__, 'device:', get_device())"
echo "
Done. Put models in $ROOT_PATH/models/{checkpoints,diffusion_models,text_encoders,vae}
(or point EXTRA_MODEL_PATHS in .env at an existing ComfyUI models folder), then run: bash start.sh"
