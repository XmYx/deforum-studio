import os

import torch
from torch import nn
from torch.cuda import nvtx

from deforum.optimizations.trt.utilities import Engine
from deforum.utils.constants import config
from deforum.utils.logging_config import logger

class TrtUnet(nn.Module):
    def __init__(
        self, use_checkpoint= False,
                image_size= 32,
                out_channels= 4,
                use_spatial_transformer= True,
                legacy= False,
                num_classes= 'sequential',
                adm_in_channels= 2816,
                dtype= torch.bfloat16,
                in_channels= 4,
                model_channels= 320,
                num_res_blocks= 2,
                attention_resolutions= [2,4],
                transformer_depth= [0,2,10],
                channel_mult= [1,2,4],
                transformer_depth_middle= 10,
                use_linear_in_transformer= True,
                context_dim= 2048,
                num_heads= -1,
                num_head_channels= 64,
                device = 'cuda',
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # self.configs = configs
        self.stream = None
        # self.model_name = model_name
        # self.lora_path = lora_path
        self.engine_vram_req = 0

        #self.loaded_config = self.configs[0]
        self.shape_hash = 0
        #self.engine = Engine(os.path.join(config.model_dir, "Unet-trt/unet.trt"))
        self.dtype = torch.float16
        self.lora_path = None

        #self.engine.activate()

    def forward(self, x, timesteps, context, *args, **kwargs):
        nvtx.range_push("forward")
        feed_dict = {
            "sample": x.float(),
            "timesteps": timesteps.float(),
            "encoder_hidden_states": context.float(),
        }
        if "y" in kwargs:
            feed_dict["y"] = kwargs["y"].float()

        # Need to check compatability on the fly
        if self.shape_hash != hash(x.shape):
            nvtx.range_push("switch_engine")
            if x.shape[-1] % 8 or x.shape[-2] % 8:
                raise ValueError(
                    "Input shape must be divisible by 64 in both dimensions."
                )
            self.switch_engine(feed_dict)
            self.shape_hash = hash(x.shape)
            nvtx.range_pop()

        tmp = torch.empty(
            self.engine_vram_req, dtype=torch.uint8, device="cuda"
        )
        self.engine.context.device_memory = tmp.data_ptr()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.engine.allocate_buffers(feed_dict)

        out = self.engine.infer(feed_dict, self.cudaStream)["latent"]

        nvtx.range_pop()
        return out

    def switch_engine(self, feed_dict):
        # valid_models, distances = modelmanager.get_valid_models(
        #     self.model_name, feed_dict
        # )
        # if len(valid_models) == 0:
        #     raise ValueError(
        #         "No valid profile found. Please go to the TensorRT tab and generate an engine with the necessary profile. If using hires.fix, you need an engine for both the base and upscaled resolutions. Otherwise, use the default (torch) U-Net."
        #     )
        #
        # best = valid_models[np.argmin(distances)]
        # if best["filepath"] == self.loaded_config["filepath"]:
        #     return
        self.deactivate()
        #FOR COG:
        # self.engine = Engine(os.path.join(os.getcwd(), "models/unet.trt"))

        #FOR PIP
        self.engine = Engine(os.path.join(config.model_dir, "unet.trt"))
        self.activate()
        self.loaded_config = "TRT"

    def activate(self):
        self.engine.load()

        self.engine_vram_req = self.engine.engine.device_memory_size
        self.engine.activate(True)

        if self.lora_path is not None:
            self.engine.refit_from_dump(self.lora_path)

    def deactivate(self):
        self.shape_hash = 0
        if hasattr(self, "engine"):
            del self.engine


# comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel = TrtUnet

logger.info("Comfy Unet HiJacked")