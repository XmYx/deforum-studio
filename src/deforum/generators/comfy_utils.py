import os
import subprocess
import sys

import torch
import torchsde

from deforum.utils.constants import comfy_path, root_path

comfy_submodules = [
    "https://github.com/XmYx/ComfyUI-AnimateDiff-Evolved",
    "https://github.com/FizzleDorf/ComfyUI_FizzNodes"
]

comfy_submodule_folders = [url.split("/")[-1] for url in comfy_submodules]

comfy_submodule_folder = os.path.join(comfy_path, "custom_nodes")


def ensure_comfy():
    # current_folder = os.getcwd()
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

    os.chdir(comfy_submodule_folder)
    for module in comfy_submodules:

        subprocess.run(["git", "clone", module])
    os.chdir(current_folder)
    sys.path.append(comfy_path)
    for path in comfy_submodule_folders:
        sys.path.append(os.path.join(comfy_submodule_folder, path))

    from comfy.cli_args import LatentPreviewMethod as lp
    class MockCLIArgsModule:
        args = mock_args
        LatentPreviewMethod = lp

    # Add the mock module to sys.modules under the name 'comfy.cli_args'
    sys.modules['comfy.cli_args'] = MockCLIArgsModule()
    import comfy.k_diffusion.sampling
    comfy.k_diffusion.sampling.BatchedBrownianTree = DeforumBatchedBrownianTree



from collections import namedtuple


# Define the namedtuple structure based on the properties identified
CLIArgs = namedtuple('CLIArgs', [
    'cpu',
    'normalvram',
    'lowvram',
    'novram',
    'highvram',
    'gpu_only',
    'disable_xformers',
    'use_pytorch_cross_attention',
    'use_split_cross_attention',
    'use_quad_cross_attention',
    'fp16_vae',
    'bf16_vae',
    'fp32_vae',
    'force_fp32',
    'force_fp16',
    'disable_smart_memory',
    'disable_ipex_optimize',
    'listen',
    'port',
    'enable_cors_header',
    'extra_model_paths_config',
    'output_directory',
    'temp_directory',
    'input_directory',
    'auto_launch',
    'disable_auto_launch',
    'cuda_device',
    'cuda_malloc',
    'disable_cuda_malloc',
    'dont_upcast_attention',
    'bf16_unet',
    'directml',
    'preview_method',
    'dont_print_server',
    'quick_test_for_ci',
    'windows_standalone_build',
    'disable_metadata'
])

# Update the mock args object with default values for the new properties
mock_args = CLIArgs(
    cpu=False,
    normalvram=False,
    lowvram=False,
    novram=False,
    highvram=True,
    gpu_only=True,
    disable_xformers=True,
    use_pytorch_cross_attention=True,
    use_split_cross_attention=False,
    use_quad_cross_attention=False,
    fp16_vae=True,
    bf16_vae=False,
    fp32_vae=False,
    force_fp32=False,
    force_fp16=False,
    disable_smart_memory=False,
    disable_ipex_optimize=False,
    listen="127.0.0.1",
    port=8188,
    enable_cors_header=None,
    extra_model_paths_config=None,
    output_directory=root_path,
    temp_directory=None,
    input_directory=None,
    auto_launch=False,
    disable_auto_launch=True,
    cuda_device=0,
    cuda_malloc=False,
    disable_cuda_malloc=True,
    dont_upcast_attention=True,
    bf16_unet=False,
    directml=None,
    preview_method="none",
    dont_print_server=True,
    quick_test_for_ci=False,
    windows_standalone_build=False,
    disable_metadata=False
)

class DeforumBatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs) for s in seed]
        else:
            self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

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
                [tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device) for tree in self.trees]) * (
                            self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)

        return w if self.batched else w[0]