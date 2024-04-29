import contextlib
import os
import subprocess
import sys
from collections import namedtuple

import torch
import torchsde

from deforum.generators.rng_noise_generator import randn_local
from deforum.utils.constants import comfy_path, root_path
from deforum.utils.logging_config import logger
from deforum.utils.string_utils import str_to_bool




def replace_torchsde_browinan():
    import torchsde._brownian.brownian_interval

    def torchsde_randn(size, dtype, device, seed):
        return randn_local(seed, size).to(device=device, dtype=dtype)

    torchsde._brownian.brownian_interval._randn = torchsde_randn



@contextlib.contextmanager
def change_dir(destination):
    cwd = os.getcwd()
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)


def clone_repo(repo_url):
    try:
        subprocess.run(["git", "clone", repo_url])
    except Exception as e:
        logger.error(f"An error occurred while cloning: {e}")


def clone_repo_to(repo_url, dest_path):
    try:
        subprocess.run(["git", "clone", repo_url, dest_path])
    except Exception as e:
        logger.error(f"An error occurred while cloning: {e}")


def add_to_sys_path(path):
    sys.path.append(path)


def ensure_comfy(custom_path=None):
    comfy_submodules = [
        "https://github.com/XmYx/ComfyUI-AnimateDiff-Evolved",
        "https://github.com/FizzleDorf/ComfyUI_FizzNodes",
    ]

    comfy_submodule_folders = [url.split("/")[-1] for url in comfy_submodules]

    comfy_path = os.environ.get("COMFY_PATH")
    comfy_submodule_folder = os.path.join(comfy_path, "custom_nodes")
    if custom_path is not None:
        comfy_path = custom_path
    else:

        comfy_path = os.environ.get("COMFY_PATH")

    comfy_update = str_to_bool(os.environ.get("COMFY_UPDATE", "False"))
        
    comfy_submodule_folder = os.path.join(comfy_path, "custom_nodes")

    if not os.path.exists(comfy_path):
        # Clone the comfy repository if it doesn't exist
        clone_repo_to("https://github.com/comfyanonymous/ComfyUI", comfy_path)
    elif comfy_update:
        # If comfy directory exists, update it.
        with change_dir(comfy_path):
            subprocess.run(["git", "pull"])

        with change_dir(comfy_submodule_folder):
            for module in comfy_submodules:
                clone_repo(module)

    # Add paths to sys.path
    add_to_sys_path(comfy_path)
    for path in comfy_submodule_folders:
        add_to_sys_path(os.path.join(comfy_submodule_folder, path))

    # Create and add the mock module to sys.modules
    from comfy.cli_args import LatentPreviewMethod as lp

    class MockCLIArgsModule:
        args = mock_args
        LatentPreviewMethod = lp

    sys.modules["comfy.cli_args"] = MockCLIArgsModule()
    #import comfy.k_diffusion.sampling
    replace_torchsde_browinan()
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    #
    # # Creating an instance of PromptServer with the loop
    # server_instance = server.PromptServer(loop)
    # execution.PromptQueue(server_instance)
    # init_custom_nodes()

    #comfy.k_diffusion.sampling.BatchedBrownianTree = DeforumBatchedBrownianTree


# Define the namedtuple structure based on the properties identified
CLIArgs = namedtuple(
    "CLIArgs",
    [
        "cpu",
        "normalvram",
        "lowvram",
        "novram",
        "highvram",
        "gpu_only",
        "disable_xformers",
        "use_pytorch_cross_attention",
        "use_split_cross_attention",
        "use_quad_cross_attention",
        "fp16_unet",
        "fp8_e4m3fn_unet",
        "fp8_e5m2_unet",
        "fp8_e4m3fn_text_enc",
        "fp8_e5m2_text_enc",
        "fp16_text_enc",
        "fp32_text_enc",
        "fp16_vae",
        "bf16_vae",
        "fp32_vae",
        "force_fp32",
        "force_fp16",
        "cpu_vae",
        "disable_smart_memory",
        "disable_ipex_optimize",
        "listen",
        "port",
        "enable_cors_header",
        "extra_model_paths_config",
        "output_directory",
        "temp_directory",
        "input_directory",
        "auto_launch",
        "disable_auto_launch",
        "cuda_device",
        "cuda_malloc",
        "disable_cuda_malloc",
        "dont_upcast_attention",
        "bf16_unet",
        "directml",
        "preview_method",
        "dont_print_server",
        "quick_test_for_ci",
        "windows_standalone_build",
        "disable_metadata",
        'deterministic',
        'multi_user',
        'max_upload_size'
    ],
)

# Update the mock args object with default values for the new properties
mock_args = CLIArgs(
    cpu=False,
    normalvram=False,
    lowvram=False,
    novram=False,
    highvram=True,
    gpu_only=False,
    disable_xformers=False,
    use_pytorch_cross_attention=True,
    use_split_cross_attention=False,
    use_quad_cross_attention=False,
    bf16_unet=False,
    fp16_unet=True,
    fp8_e4m3fn_unet=False,
    fp8_e5m2_unet=False,
    fp8_e4m3fn_text_enc=False,
    fp8_e5m2_text_enc=False,
    fp16_text_enc=True,
    fp32_text_enc=False,
    fp16_vae=False,
    bf16_vae=True,
    fp32_vae=False,
    force_fp32=False,
    force_fp16=False,
    cpu_vae=False,
    disable_smart_memory=False,
    disable_ipex_optimize=False,
    listen="127.0.0.1",
    port=8188,
    enable_cors_header=None,
    extra_model_paths_config="config/comfy_paths.yaml",
    output_directory=root_path,
    temp_directory=None,
    input_directory=None,
    auto_launch=False,
    disable_auto_launch=True,
    cuda_device=0,
    cuda_malloc=False,
    disable_cuda_malloc=False,
    dont_upcast_attention=False,
    directml=None,
    preview_method="none",
    dont_print_server=True,
    quick_test_for_ci=False,
    windows_standalone_build=False,
    disable_metadata=False,
    deterministic=False,
    multi_user=True,
    max_upload_size=1024
)


class DeforumBatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [
                torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs)
                for s in seed
            ]
        else:
            self.trees = [
                torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed
            ]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        if torch.abs(t0 - t1) < 1e-6:  # or some other small value
            # Handle this case, e.g., return a zero tensor of appropriate shape
            return torch.zeros_like(t0)
        if self.cpu_tree:
            w = torch.stack(
                [
                    tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device)
                    for tree in self.trees
                ]
            ) * (self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)

        return w if self.batched else w[0]
