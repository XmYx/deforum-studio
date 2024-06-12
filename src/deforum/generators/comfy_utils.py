"""
This module provides utilities for managing and configuring ComfyUI extensions,
including downloading models, cloning repositories, and handling custom nodes.
"""
import contextlib
import importlib
import logging
import os
import subprocess
import sys
import traceback
from collections import namedtuple

import requests
import torch
import torchsde

from deforum.generators.rng_noise_generator import randn_local
from deforum.utils.constants import config
from deforum.utils.logging_config import logger

ip_adapter_model_dict = {
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors": f"{config.comfy_path}/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors": f"{config.comfy_path}/models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors"
}

comfy_initialized = False
cfg_guider = None

def replace_torchsde_browinan():
    """
    Replace the default torchsde Brownian noise generator with a custom one using local random noise.
    """
    import torchsde._brownian.brownian_interval
    def torchsde_randn(size, dtype, device, seed):
        return randn_local(seed, size).to(device=device, dtype=dtype)
    torchsde._brownian.brownian_interval._randn = torchsde_randn

@contextlib.contextmanager
def change_dir(destination):
    """
    Context manager to temporarily change the current working directory.

    Args:
        destination (str): The path to change to.
    """
    cwd = os.getcwd()
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)


def clone_repo(repo_url, commit_id=None):
    """
    Clone a Git repository and optionally check out a specific commit.

    Args:
        repo_url (str): The URL of the repository to clone.
        commit_id (str, optional): The commit ID to check out. Defaults to None.
    """
    try:
        subprocess.run(["git", "clone", repo_url])
        if commit_id:
            repo_name = repo_url.split("/")[-1].split(".")[0]
            os.chdir(repo_name)
            subprocess.run(["git", "checkout", commit_id])
            os.chdir("..")
    except Exception as e:
        logger.error(f"An error occurred while cloning: {e}")


def clone_repo_to(repo_url, dest_path):
    """
    Clone a Git repository to a specific destination path.

    Args:
        repo_url (str): The URL of the repository to clone.
        dest_path (str): The destination path to clone the repository to.
    """
    try:
        subprocess.run(["git", "clone", repo_url, dest_path])
    except Exception as e:
        logger.error(f"An error occurred while cloning: {e}")


def add_to_sys_path(path):
    """
    Add a directory to the system path.

    Args:
        path (str): The directory path to add.
    """
    sys.path.append(path)

def download_models(model_dict):
    """
    Download models from given URLs and save them to specified local paths.

    Args:
        model_dict (dict): A dictionary with URLs as keys and local paths as values.
    """
    for url, local_path in model_dict.items():
        # Check if the local directory exists, create it if it doesn't
        local_dir = os.path.dirname(local_path)
        os.makedirs(local_dir, exist_ok=True)

        # Check if the file already exists
        if not os.path.exists(local_path):
            print(f"Downloading {url} to {local_path}...")
            try:
                # Download the file
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Save the file
                with open(local_path, 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded {local_path} successfully.")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {url}: {e}")
        else:
            print(f"File {local_path} already exists. Skipping download.")
loaded_objects = {}
last_helds = {}
def globals_cleanup(prompt):
    pass
def ensure_comfy(custom_path=None):
    """
    Add 'ComfyUI' to the sys.path
    """
    global comfy_initialized
    if not comfy_initialized:
        comfy_initialized = True

        curr_folder = os.getcwd()
        comfy_submodules = [
            ('https://github.com/ltdrdata/ComfyUI-Impact-Pack', '48d9ce7528f83074b6db7a7b15ef7e88c7134aa5'),
            ('https://github.com/XmYx/ComfyUI-Inspire-Pack', 'd40389f93d6f42b44e0e2f02190a216762d028d8'),
            ('https://github.com/shiimizu/ComfyUI_smZNodes', 'a1627ce2ade31822694d82aa9600a4eff0f99d69'),
            ('https://github.com/gameltb/ComfyUI_stable_fast', 'c0327e6f076bd8a36e3c29f3594025c76cf9beae'),
            ('https://github.com/cubiq/ComfyUI_IPAdapter_plus', '20125bf9394b1bc98ef3228277a31a3a52c72fc2')
        ]
        comfy_path = custom_path or config.comfy_path
        comfy_submodule_folder = os.path.join(comfy_path, 'custom_nodes')

        # if not os.path.exists(config.comfy_path):
        try:
            print("Initializing and updating git submodules...")
            subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])
            print("Submodules initialized and updated.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to initialize submodules: {e}")

        with change_dir(comfy_submodule_folder):
            for module, commit_id in comfy_submodules:
                if not os.path.exists(os.path.join(os.getcwd(), module.split("/")[-1])):
                    clone_repo(module, commit_id)

        os.chdir(curr_folder)
        download_models(ip_adapter_model_dict)
        if comfy_path is not None and os.path.isdir(comfy_path):
            sys.path.append(comfy_path)
            print(f"'{comfy_path}' added to sys.path")

        import asyncio
        import execution
        from nodes import init_custom_nodes
        import server

        # Creating a new event loop and setting it as the default loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Creating an instance of PromptServer with the loop
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)

        # Initializing custom nodes
        init_custom_nodes()

        def replace_function(module_path, function_name, new_function):
            """
            Replace a function in a given module with a new function.

            Args:
                module_path (str): The file path of the module.
                function_name (str): The name of the function to replace.
                new_function (callable): The new function to replace the old one with.
            """
            # Load the module dynamically
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Replace the function in the module
            setattr(module, function_name, new_function)
            print(f"Function {function_name} in {module_path} replaced successfully.")

            # Ensure the replacement persists in all references
            module_names = list(sys.modules.keys()) # Avoid modifyin sys.modules dictionary while iterating over it
            for name in module_names:
                mod = sys.modules.get(name)
                if mod and hasattr(mod, function_name):
                    setattr(mod, function_name, new_function)
                    print(f"Function {function_name} in module {name} replaced successfully.")

        # Define the new implementation of globals_cleanup
        def new_globals_cleanup(*args, **kwargs):
            # New implementation of the function
            print("New globals_cleanup function executed")

        # Path to the module
        module_path = os.path.join(comfy_submodule_folder, 'efficiency-nodes-comfyui/tsc_utils.py')

        # Replace the function
        replace_function(module_path, 'globals_cleanup', new_globals_cleanup)

class CLIArgs:
    def __init__(self):
        self.cpu = False
        self.normalvram = False
        self.lowvram = False
        self.novram = False
        self.highvram = True
        self.gpu_only = False
        self.disable_xformers = False
        self.use_pytorch_cross_attention = True
        self.use_split_cross_attention = False
        self.use_quad_cross_attention = False
        self.bf16_unet = False
        self.fp16_unet = True
        self.fp8_e4m3fn_unet = False
        self.fp8_e5m2_unet = False
        self.fp8_e4m3fn_text_enc = False
        self.fp8_e5m2_text_enc = False
        self.fp16_text_enc = True
        self.fp32_text_enc = False
        self.fp16_vae = True
        self.bf16_vae = False
        self.fp32_vae = False
        self.force_fp32 = False
        self.force_fp16 = False
        self.cpu_vae = False
        self.disable_smart_memory = True
        self.disable_ipex_optimize = False
        self.listen = "127.0.0.1"
        self.port = 8188
        self.enable_cors_header = None
        self.extra_model_paths_config = "config/comfy_paths.yaml"
        self.output_directory = config.output_dir
        self.temp_directory = None
        self.input_directory = None
        self.auto_launch = False
        self.disable_auto_launch = True
        self.cuda_device = 0
        self.cuda_malloc = False
        self.disable_cuda_malloc = True
        self.force_upcast_attention = False
        self.dont_upcast_attention = False
        self.directml = None
        self.preview_method = "none"
        self.dont_print_server = True
        self.quick_test_for_ci = False
        self.windows_standalone_build = False
        self.disable_metadata = False
        self.deterministic = False
        self.multi_user = True
        self.max_upload_size = 1024

mock_args = CLIArgs()
class HIJackCFGGuider:
    def __init__(self, model_patcher):
        # print("BIG HOOOORAAAAY\n\n\n\n\n")
        self.model_patcher = model_patcher
        self.inner_model = self.model_patcher.model
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        import comfy

        for k in conds:
            self.original_conds[k] = comfy.sampler_helpers.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        import comfy

        return comfy.samplers.sampling_function(
            self.model_patcher.model,
            x,
            timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            self.cfg,
            model_options=model_options,
            seed=seed,
        )

    def inner_sample(
        self,
        noise,
        latent_image,
        device,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
    ):
        import comfy

        if (
            latent_image is not None and torch.count_nonzero(latent_image) > 0
        ):  # Don't shift the empty latent image.
            latent_image = self.model_patcher.model.process_latent_in(latent_image)

        self.conds = comfy.samplers.process_conds(
            self.model_patcher.model,
            noise,
            self.conds,
            device,
            latent_image,
            denoise_mask,
            seed,
        )

        extra_args = {"model_options": self.model_options, "seed": seed}

        samples = sampler.sample(
            self,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
        return self.model_patcher.model.process_latent_out(samples.to(torch.float16))

    def sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        self.conds = self.original_conds
        device = torch.device("cuda")
        sigmas = sigmas.to(device)
        output = self.inner_sample(
            noise, latent_image, device, sampler, sigmas, denoise_mask, None, True, seed
        )
        return output


def sampleDeforum(
    model,
    noise,
    positive,
    negative,
    cfg,
    device,
    sampler,
    sigmas,
    model_options={},
    latent_image=None,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    global cfg_guider
    if cfg_guider is None:
        cfg_guider = HIJackCFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(
        noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed
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
