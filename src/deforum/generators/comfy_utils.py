"""
Bootstraps ComfyUI as a library.

Deforum uses ComfyUI's model loading, text encoders, VAEs and samplers directly (no server, no
graph execution), which gives it every model family ComfyUI supports: SD 1.x/2.x, SDXL, SD3,
FLUX.1/2/3, Chroma, Qwen-Image and friends, on CUDA, ROCm, MPS, XPU or CPU.

Model search paths, in order:
  * ``<ROOT_PATH>/models/<kind>`` (e.g. ``~/deforum/models/checkpoints``) and ``<ROOT_PATH>/models``
  * the bundled ComfyUI's own ``models`` folder
  * every ComfyUI ``models`` root listed in ``EXTRA_MODEL_PATHS`` (os.pathsep separated), and the
    entries of a ComfyUI ``extra_model_paths.yaml`` given by ``EXTRA_MODEL_PATHS_YAML``.
"""
import os
import subprocess
import sys

from deforum.utils.constants import config
from deforum.utils.logging_config import logger

comfy_initialized = False

# ComfyUI folder names Deforum looks into, and the sub folders of a models root that map to them.
MODEL_KINDS = {
    "checkpoints": ["checkpoints"],
    "diffusion_models": ["diffusion_models", "unet"],
    "text_encoders": ["text_encoders", "clip"],
    "vae": ["vae"],
    "loras": ["loras"],
    "embeddings": ["embeddings"],
    "clip_vision": ["clip_vision"],
    "controlnet": ["controlnet"],
}


def _register_models_root(folder_paths, root, label, flat_checkpoints=False):
    """Register <root>/<kind> folders; ``flat_checkpoints`` also treats files directly in root as checkpoints."""
    if not root or not os.path.isdir(root):
        return
    if flat_checkpoints:
        folder_paths.add_model_folder_path("checkpoints", root)
    for kind, subdirs in MODEL_KINDS.items():
        for sub in subdirs:
            path = os.path.join(root, sub) if sub else root
            if os.path.isdir(path):
                folder_paths.add_model_folder_path(kind, path)
    logger.info(f"[comfy] model root ({label}): {root}")


def _init_submodule(comfy_path):
    if os.path.isfile(os.path.join(comfy_path, "comfy", "sd.py")):
        return
    logger.info("ComfyUI not found, initializing the git submodule...")
    repo_root = os.path.dirname(os.path.abspath(config.src_path))
    try:
        subprocess.check_call(["git", "submodule", "update", "--init", "src/ComfyUI"], cwd=repo_root)
    except Exception as e:
        raise RuntimeError(
            f"ComfyUI is missing at {comfy_path}. Run `git submodule update --init` or set COMFY_PATH."
        ) from e


def _select_comfy_device():
    """Tell ComfyUI to run on the CPU when no accelerator is usable (or DEFORUM_DEVICE=cpu).

    ComfyUI otherwise assumes CUDA whenever torch is a CUDA build, even with no visible GPU."""
    import torch

    import comfy.cli_args
    forced = os.environ.get("DEFORUM_DEVICE", "").lower()
    has_accel = torch.cuda.is_available() or \
        (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()) or \
        (hasattr(torch, "xpu") and torch.xpu.is_available())
    if forced == "cpu" or not has_accel:
        comfy.cli_args.args.cpu = True
    elif forced.startswith("cuda:"):
        comfy.cli_args.args.cuda_device = int(forced.split(":", 1)[1])


def ensure_comfy(custom_path=None):
    """Make ComfyUI importable and register Deforum's model folders. Safe to call repeatedly."""
    global comfy_initialized
    if comfy_initialized:
        return
    comfy_path = os.path.abspath(custom_path or config.comfy_path)
    _init_submodule(comfy_path)
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)

    # Never let ComfyUI parse Deforum's command line.
    import comfy.options
    comfy.options.args_parsing = False

    _select_comfy_device()
    import comfy.model_management  # noqa: F401  (selects the device)
    import folder_paths

    os.makedirs(config.model_dir, exist_ok=True)
    # older Deforum versions downloaded checkpoints straight into ~/deforum/models
    _register_models_root(folder_paths, config.model_dir, "deforum", flat_checkpoints=True)
    from decouple import config as env  # environment first, then .env / settings.ini
    for root in filter(None, env("EXTRA_MODEL_PATHS", default="").split(os.pathsep)):
        _register_models_root(folder_paths, os.path.expanduser(root), "extra")
    yaml_path = env("EXTRA_MODEL_PATHS_YAML", default="")
    if yaml_path and os.path.isfile(yaml_path):
        from utils.extra_config import load_extra_path_config
        load_extra_path_config(yaml_path)

    from deforum.utils.device import get_device, reset_device_cache
    reset_device_cache()
    _enable_fast_math(env)
    comfy_initialized = True
    logger.info(f"[comfy] ComfyUI ready from {comfy_path} on device {get_device()}")


def _enable_fast_math(env):
    """fp16 matmul accumulation: ~15-20% faster sampling on NVIDIA (DEFORUM_FP16_ACCUMULATION=0 to disable)."""
    import torch
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return
    if env("DEFORUM_FP16_ACCUMULATION", default=True, cast=bool) and \
            hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


_custom_nodes_loaded = False


def load_comfy_custom_nodes():
    """Load ComfyUI's extra + custom nodes (only needed for optional features such as IP-Adapter)."""
    global _custom_nodes_loaded
    ensure_comfy()
    if _custom_nodes_loaded:
        return
    import asyncio

    import execution
    import nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
    _custom_nodes_loaded = True
