import json
import os
import sys
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from ..deforum_pipeline import DeforumBase
from deforum.pipeline_utils import DeforumGenerationObject, extract_values
from deforum.utils.constants import config
from deforum.utils.video_save_util import save_as_h264
from deforum.utils.logging_config import logger

def generate_keyframe_sequence(prompts, max_frames):
    keyframe_sequence = {}

    # Calculate the interval for each prompt based on the number of prompts and max_frames
    interval = max_frames // (len(prompts) - 1)

    for index, prompt in enumerate(prompts):
        frame_number = index * interval

        # If it's the last prompt, ensure it's assigned to the last frame
        if index == len(prompts) - 1:
            frame_number = max_frames

        keyframe_sequence[frame_number] = prompt

    return keyframe_sequence
def parse_weight_string(weight_string, max_frames):
    # If empty string, return list with all 1.0 values
    if not weight_string:
        return [1.0] * max_frames

    # Parse the string
    parts = weight_string.split(',')
    keyframe_dict = {}
    for part in parts:
        key, value = part.split(':')
        key = int(key.strip())
        value = float(value.strip('() '))
        keyframe_dict[key] = value

    # Create a series with NaN values
    series = pd.Series([np.nan] * max_frames)

    # Set the parsed values
    for key, value in keyframe_dict.items():
        series[key] = value

    # Linearly interpolate missing values and fill any remaining NaNs with the backfill method
    series.interpolate(method='linear', inplace=True)
    series.fillna(method='bfill', inplace=True)
    series.fillna(method='ffill', inplace=True)

    return series.tolist()

class DeforumAnimateDiffPipeline(DeforumBase):

    def __init__(self, generator: Callable, logger: Optional[Callable] = None):

        super().__init__()
        self.generator = generator
        self.logger = logger
        self.animatediff_path = os.path.join(config.comfy_path, "custom_nodes", "ComfyUI-AnimateDiff-Evolved")
        sys.path.append(self.animatediff_path)
        from animatediff import sampling

        import comfy.sample as comfy_sample
        # override comfy_sample.sample with animatediff-support version
        comfy_sample.sample = sampling.animatediff_sample_factory(comfy_sample.sample)

        self.prep_fns = []
        self.shoot_fns = []
        self.post_fns = []


    def __call__(self, settings_file:Optional[str] = None, *args, **kwargs):
        """
        AnimateDiff / HotshotCo-XL Sampler function with sliding context.

        This method initializes a DeforumGenerationObject with either default settings or settings loaded from a given file.
        It then updates the generation parameters based on provided keyword arguments and creates a sliding context
        for animation or HotshotCo-XL generation.

        Args:
            settings_file (Optional[str]): Path to a .txt settings file to load. Defaults to None, which will load default settings.

        Keyword Args:
            context_length (int): Length of the context for the animation sequence. Defaults to 16.
            context_stride (int): Stride of the context for the animation sequence. Defaults to 1.
            context_overlap (int): Overlap of the context in the animation sequence. Defaults to 4.
            max_frames (int): Maximum number of frames for the animation sequence. Defaults to 32.
            closed_loop (bool): Whether the animation should loop back to the start. Defaults to True.
            Any additional keyword arguments will be passed to the update_from_kwargs method.
            sampler_name: ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm",
                   "ddim", "uni_pc", "uni_pc_bh2"]
            scheduler: ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
            steps (int): 25
            scale (float): 7.5

        Default Values:
            - context_length: 16 (range: 0-32)
            - context_stride: 1 (range: 1-32)
            - context_overlap: 4 (range: 0-32)
            - max_frames: 32 (range: 1-4096)
            - closed_loop: True

        Prompts:
            A dictionary mapping frame numbers to text prompts for keyframes in the animation sequence.
            Example:
                {
                    0: "Enchanted forest, beautiful trees and butterflies flying around",
                    32: "Abstract art of an enchanted forest with butterflies"
                }

        Prompt List:
            A list of strings that will be converted into a keyframe sequence if provided. The 'generate_keyframe_sequence'
            function will be used to distribute these prompts evenly across the animation sequence.

        Returns:
            None. This method operates by side effects, updating the internal state of the DeforumGenerationObject instance.

        """
        if settings_file:
            self.gen = DeforumGenerationObject.from_settings_file(settings_file)
        else:
            self.gen = DeforumGenerationObject()


        self.gen.seed = -1
        self.gen.scale = 7.5
        self.gen.steps = 25
        #Create default values needed for animatediff / Hotshotco-XL generation
        self.gen.prompts = json.loads(json.dumps(animatediff_default_prompts))

        defaults = extract_values(animate_diff_defaults)
        #Update generation params from any kwarg passed to the call
        self.gen.update_from_kwargs(prompts=animatediff_default_prompts, **defaults)
        self.gen.update_from_kwargs(**kwargs)
        if isinstance(self.gen.prompts, str):
            self.gen.prompts = {0:str(self.gen.prompts),
                                self.gen.max_frames:str(self.gen.prompts)}
        if hasattr(self.gen, "prompt_list"):
            try:
                seq = generate_keyframe_sequence(self.gen.prompt_list, self.gen.max_frames)
                self.gen.prompts = seq
            except:
                pass



        #Create 'sliding context' for animatediff / Hotshotco-XL
        #TODO Move functions in the the loops as and if they make sense

        latent = kwargs.get('latent')
        _ = self.run_animatediff(latent=latent)

        return self.gen

    def create_context(self, version="1.5"):
        from custom_nodes.ComfyUI_FizzNodes import PromptScheduleEncodeSDXL, BatchPromptSchedule, BatchPromptScheduleLatentInput
        from animatediff.context import UniformContextOptions
        self.context_options = UniformContextOptions(
            context_length=self.gen.context_length, # ("INT", {"default": 16, "min": 1, "max": 32})
            context_stride=self.gen.context_stride, # ("INT", {"default": 1, "min": 1, "max": 32})
            context_overlap=self.gen.context_overlap, # ("INT", {"default": 4, "min": 0, "max": 32})
            context_schedule="uniform", #(ContextSchedules.CONTEXT_SCHEDULE_LIST,)
            closed_loop=self.gen.closed_loop, #("BOOLEAN", {"default": False},)
            )

        if version == "XL":
            prompt_scheduler = PromptScheduleEncodeSDXL()

            self.conds = prompt_scheduler.animate(clip=self.generator.clip,
                                                  width=self.gen.width,
                                                  height=self.gen.height,
                                                  crop_w=0,
                                                  crop_h=0,
                                                  target_width=self.gen.width,
                                                  target_height=self.gen.height,
                                                  text_g=json.dumps(self.gen.prompts).strip("{}"),
                                                  text_l=json.dumps(self.gen.prompts).strip("{}"),
                                                  max_frames=self.gen.max_frames,
                                                  current_frame=0,
                                                  pre_text_L="Abstract art",
                                                  pre_text_G="",
                                                  app_text_L="",
                                                  app_text_G="",
                                                  pw_a=[1.0 for _ in range(self.gen.max_frames)],#parse_weight_string("", self.gen.max_frames),
                                                  pw_b=[1.0 for _ in range(self.gen.max_frames)],#parse_weight_string("", self.gen.max_frames),
                                                  pw_c=[1.0 for _ in range(self.gen.max_frames)],#parse_weight_string("", self.gen.max_frames),
                                                  pw_d=[1.0 for _ in range(self.gen.max_frames)],#parse_weight_string("", self.gen.max_frames),
                                                  )[0]
        else:
            prompt_scheduler = BatchPromptScheduleLatentInput()

            from comfy import model_management

            def maximum_batch_area():
                from comfy.model_management import get_free_memory
                memory_free = get_free_memory() / (1024 * 1024)
                area = ((memory_free - 1024) * 0.9) / (0.6)
                return int(max(area, 0))

            model_management.maximum_batch_area = maximum_batch_area
            # self.conds = self.generator.get_conds(self.gen.prompt)

            assert self.gen.max_frames % 16 == 0, "Make sure to pass an animatediff max frames value that is a multiple of 16"
            self.latents = {"samples":torch.randn([self.gen.max_frames, 4, self.gen.height // 8, self.gen.width // 8])}
            logger.info(f"Prompts: {self.gen.prompts}")
            self.conds, self.n_conds, self.latents = prompt_scheduler.animate(text=json.dumps(self.gen.prompts).strip("{}"),
                                                                             num_latents=self.latents,
                                                                             print_output=True,
                                                                             clip=self.generator.clip,
                                                                              start_frame=0,
                                                                             pw_a=parse_weight_string("", self.gen.max_frames),
                                                                             pw_b=parse_weight_string("", self.gen.max_frames),
                                                                             pw_c=parse_weight_string("", self.gen.max_frames),
                                                                             pw_d=parse_weight_string("", self.gen.max_frames),
                                                                             )
            from custom_nodes.PPF_Noise_ComfyUI.nodes import PPFNoiseNode
            noise_generator = PPFNoiseNode()
            """
                        "required": {
                "batch_size": ("INT", {"default": 1, "max": 64, "min": 1, "step": 1}),
                "width": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "height": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "resampling": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
                "X": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "Y": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "Z": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "evolution": ("FLOAT", {"default": 0.0, "max": 1.0, "min": 0.0, "step": 0.01}),
                "frame": ("INT", {"default": 0, "max": 99999999, "min": 0, "step": 1}),
                "scale": ("FLOAT", {"default": 5, "max": 2048, "min": 2, "step": 0.01}),
                "octaves": ("INT", {"default": 8, "max": 8, "min": 1, "step": 1}),
                "persistence": ("FLOAT", {"default": 1.5, "max": 23.0, "min": 0.01, "step": 0.01}),
                "lacunarity": ("FLOAT", {"default": 2.0, "max": 99.0, "min": 0.01, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 4.0, "max": 38.0, "min": 0.01, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.01}),
                "clamp_min": ("FLOAT", {"default": 0.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "clamp_max": ("FLOAT", {"default": 1.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
                "device": (["cpu", "cuda"],),
            },
            
            
            """
            self.latents = noise_generator.power_fractal_latent(batch_size=self.gen.max_frames,
                                                                width=self.gen.width,
                                                                height=self.gen.height,
                                                                resampling="area",
                                                                X=0.0,
                                                                Y=0.0,
                                                                Z=0.0,
                                                                evolution=0.2,
                                                                frame=0,
                                                                scale=5,
                                                                octaves=8,
                                                                persistence=1.5,
                                                                lacunarity=2.0,
                                                                exponent=4.0,
                                                                brightness=0.0,
                                                                contrast=0.0,
                                                                clamp_min=0.0,
                                                                clamp_max=1.0,
                                                                seed=self.gen.seed,
                                                                device='cuda')[0]


    @torch.inference_mode()
    def run_animatediff(self, latent=None):
        from animatediff.model_utils import BetaSchedules
        from animatediff.motion_module import load_motion_module, inject_params_into_model, InjectionParams
        from animatediff.context import UniformContextOptions
        from deforum.utils.file_dl_util import download_file_to

        from comfy import model_base

        def get_motion_model_path(model_name: str):
            return os.path.join(self.animatediff_path, 'models', model_name)

        import animatediff.motion_module
        animatediff.motion_module.get_motion_model_path = get_motion_model_path

        if isinstance(self.generator.model.model, model_base.SDXL):
            url = "https://huggingface.co/hotshotco/Hotshot-XL/resolve/main/hsxl_temporal_layers.f16.safetensors"
        elif isinstance(self.generator.model.model, model_base.BaseModel):
            url = "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"

        destination = os.path.join(self.animatediff_path, 'models')
        self.module_filename = url.split('/')[-1]
        module_path = os.path.join(destination, self.module_filename)
        if not os.path.isfile(module_path):
            module_path = download_file_to(url=url, destination_dir=destination, filename=self.module_filename)

        self.create_context()

        #TODO load motion module
        motion_lora = None

        mm = load_motion_module(self.module_filename, motion_lora, model=self.generator.model)
        # set injection params
        injection_params = InjectionParams(
                video_length=self.gen.max_frames,
                unlimited_area_hack=False,
                apply_mm_groupnorm_hack=True,
                beta_schedule=BetaSchedules.LINEAR, # beta_schedule
                injector=mm.injector_version,
                model_name=self.module_filename,
        )
        if self.context_options:
            # set context settings TODO: make this dynamic for future purposes
            if type(self.context_options) == UniformContextOptions:
                injection_params.set_context(
                        context_length=self.context_options.context_length,
                        context_stride=self.context_options.context_stride,
                        context_overlap=self.context_options.context_overlap,
                        context_schedule=self.context_options.context_schedule,
                        closed_loop=self.context_options.closed_loop
                )
        if motion_lora:
            injection_params.set_loras(motion_lora)
        # inject for use in sampling code

        self.generator.model = inject_params_into_model(self.generator.model, injection_params)

        #height = 512
        #width = 512
        # batch_size = self.gen.max_frames
        #
        # prompt = "Mona Lisa walking in new york"
        # n_prompt = ""



        if latent == None and self.latents == None:
            self.latents = torch.randn([self.gen.max_frames, 4, self.gen.height // 8, self.gen.width // 8]).to("cuda")
            strength = 1.0
        else:
            strength = 0.5
        images = self.generator( pooled_prompts=self.conds,
                                 latent=self.latents,
                                 strength=strength,
                                 steps=self.gen.steps,
                                 scale=self.gen.scale,
                                 sampler_name=self.gen.sampler_name,
                                 scheduler=self.gen.scheduler,
                                 seed=self.gen.seed)

        #self.gen.frame_interpolation_engine = "FILM"
        self.gen.frame_interpolation_x_amount = 2

        self.images = [np.array(image) for image in images]
        self.gen.images = self.images
        from deforum.pipelines.deforum_animation.animation_helpers import film_interpolate_cls

        if self.gen.frame_interpolation_engine != "None":
            film_interpolate_cls(self)

        if not self.gen.store_frames_in_ram:
            save_as_h264(self.images, f"output/video/{self.gen.batch_name}.mp4")
            self.gen.video_path = f"output/video/{self.gen.batch_name}.mp4"
            logger.info(f"ANIMATEDIFF SAVED: {self.gen.video_path}")
        return self.gen

animate_diff_defaults = {
    "context_length": {
        "label": "Context Length",
        "type": "number",
        "minimum": 0,
        "maximum": 32,
        "step": 1,
        "value": 16,
        "visible": True
    },
    "context_stride": {
        "label": "Context Stride",
        "type": "number",
        "minimum": 1,
        "maximum": 32,
        "step": 1,
        "value": 1,
        "visible": True
    },
    "context_overlap": {
        "label": "Context Overlap",
        "type": "number",
        "minimum": 0,
        "maximum": 32,
        "step": 1,
        "value": 4,
        "visible": True
    },
    "max_frames": {
        "label": "Max Frames",
        "type": "number",
        "minimum": 1,
        "maximum": 4096,
        "step": 1,
        "value": 32,
        "visible": True
    },
    "closed_loop": {
        "label": "Closed Loop",
        "type": "checkbox",
        "value": False,
        "info": "",
        "visible": True
    },

}
animatediff_default_prompts = {
    0: "Enchanted forest, beautiful trees and butterflies flying around",
    32: "Abstract art of an enchanted forest with butterflies"
}
