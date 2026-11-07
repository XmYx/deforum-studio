#!/usr/bin/env bash
# Optional: environment for the "World Model" animation mode (FLUX.3 Action).
#
# FLUX.3 Action's inference code (github.com/black-forest-labs/flux-action) pins Python 3.12 and its own
# transformers/numpy versions, so it gets a separate virtualenv; Deforum talks to it through
# src/deforum/worldmodel/flux3_action_worker.py.
#
# The weights are gated: accept the FLUX Kommunity license on
#   https://huggingface.co/black-forest-labs/flux-3-action-droid  (and -base, which holds the shared encoders)
# and log in with `hf auth login` (a fine-grained token needs "read access to public gated repos").
#
#   bash install_worldmodel.sh
# Uses flux-action's validated stack: torch 2.10.0 + CUDA 12.8 and the matching NATTEN wheel (the video VAE's
# neighborhood attention; NATTEN publishes wheels per torch/CUDA pair, see https://whl.natten.org).
# Overrides: FLUX_ACTION_REF (git ref, default pinned below)
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv-worldmodel"
REF="${FLUX_ACTION_REF:-472653e}"
TORCH="2.10.0"
TORCHVISION="0.25.0"
CUDA_TAG="cu128"
NATTEN="0.21.6+torch2100cu128"

if command -v uv >/dev/null 2>&1; then
  uv venv --allow-existing -p 3.12 "$VENV"
  PIP=(uv pip install -p "$VENV")
else
  [ -x "$VENV/bin/python" ] || python3.12 -m venv "$VENV"
  "$VENV/bin/python" -m pip install -U pip
  PIP=("$VENV/bin/python" -m pip install)
fi
"${PIP[@]}" "torch==$TORCH" "torchvision==$TORCHVISION" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
"${PIP[@]}" "flux-action[encoders] @ git+https://github.com/black-forest-labs/flux-action@$REF" pillow
"${PIP[@]}" "natten==$NATTEN" -f https://whl.natten.org/

"$VENV/bin/python" -c "from flux_action.policy import FluxActionPolicy; import natten, torch; print('flux-action ok, natten', natten.__version__, 'cuda:', torch.cuda.is_available())"
echo "
World model environment ready: $VENV
Pick 'World Model' as the animation mode. Defaults: WORLD_MODEL_POLICY=black-forest-labs/flux-3-action-droid,
WORLD_MODEL_SUBFOLDER=variants/gd-fp8r (set them in .env to use another FLUX.3 Action package)."
