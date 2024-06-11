"""
The module implements an animation pipeline for Deforum which includes a setup, main loop, and cleanup. The pipeline is
stateful due to maintaining the intermediate state throughout the process which may be CPU and memory-intensive.
It enables the loading and processing of generator function responsible for animation creation. The pipeline supports
optional logging of its internal activities for debugging and tracing purposes. Reset of the pipeline state is also supported.
"""
import gc
import math
import os
import re
import secrets
import shutil
import time
from glob import glob
from typing import Callable, Optional

import cv2
import numexpr
import numpy as np
import pandas as pd
import PIL.Image
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from tqdm import tqdm

from deforum.docutils.decorator import deforumdoc
from deforum.utils.logging_config import logger

from ... import ComfyDeforumGenerator
from ...models import RAFT, DepthModel
from ...pipeline_utils import DeforumGenerationObject
from ...utils.constants import config
from ...utils.deforum_hybrid_animation import hybrid_generation
from ...utils.deforum_logger_util import Logger
from ...utils.image_utils import load_image
from ...utils.resume_vars import get_resume_vars
from ...utils.sdxl_styles import STYLE_NAMES, apply_style
from ...utils.string_utils import split_weighted_subprompts
from ..deforum_pipeline import DeforumBase
from .animation_helpers import (
    DeforumAnimKeys,
    LooperAnimKeys,
    add_noise_cls,
    affine_persp_motion,
    anim_frame_warp_cls,
    apply_temporal_flow_cls,
    cls_subtitle_handler,
    color_match_cls,
    color_match_video_input,
    diffusion_redo,
    film_interpolate_cls,
    generate_interpolated_frames,
    get_generation_params,
    handle_noise_mask,
    hybrid_composite_cls,
    main_generate_with_cls,
    optical_flow_motion,
    optical_flow_redo,
    overlay_mask_cls,
    post_color_match_with_cls,
    post_gen_cls,
    post_hybrid_composite_cls,
    rife_interpolate_cls,
    save_video_cls,
    set_contrast_image, run_adiff_cls,
)
from .animation_params import auto_to_comfy
from .parseq_adapter import ParseqAdapter


class DeforumAnimationPipeline(DeforumBase):
    """
    Animation pipeline for Deforum.

    Provides a mechanism to run an animation generation process using the provided generator.
    Allows for pre-processing, main loop, and post-processing steps.
    Uses a logger to record the metrics and timings of each step in the pipeline.
    """
    script_start_time = time.time()

    @deforumdoc
    def __init__(self, generator: Callable, logger: Optional[Callable] = None) -> None:
        """
        Initializes the DeforumAnimationPipeline.

        :param generator Callable: The generator function for producing animations.
        :param logger Optional[Callable]: Optional logger function. If not provided, a default logger is created. Defaults to None.
        :return: None.
        :rtype: None

        Example usage:

        ``python
        generator_func = callable_generator()
        logger_func = callable_logger()

        animation_pipeline = DeforumAnimationPipeline(generator=generator_func, logger=logger_func)
        ``

        Args:
            generator (Callable): The generator function for producing animations.
            logger (Optional[Callable], optional): Optional logger function. Defaults to None.
        """
        super().__init__()

        self.generator = generator

        if logger is None:
            self.logger = Logger(config.root_path)
        else:
            self.logger = logger

        self.prep_fns = []
        self.shoot_fns = []
        self.post_fns = []
        self.images = []
        if self.logger is not None:
            self.logging = True
        else:
            self.logging = False

        self.interrupt = False
        self.raft_model = None
        self.loaded_depth_model = ""
        self.depth_model = None

    @deforumdoc
    def __call__(self, settings_file: str = None, callback=None, *args, **kwargs) -> DeforumGenerationObject:
        """
        Execute the animation pipeline.

        Args:
            :param settings_file Optional[str]: Optional path to the settings file. Defaults to None.
            :param callback Optional[Callable]: Optional callback function to be executed during pipeline. Default to None.
            :param kwargs dict: Additional arguments to be submitted to the pipeline.
            :return: DeforumGenerationObject: The generated object after the pipeline execution.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Example usage:
        ```python
        settings = '/path/to/settings/file'
        callback_func = callable_callback()

        deforum_object = animation_pipeline(settings_file=settings, callback=callback_func)
        ```
        Returns:
            DeforumGenerationObject: The generated object after the pipeline execution.
        """

        self.interrupt = False

        self.combined_pre_checks(settings_file, callback, *args, **kwargs)

        self.log_function_lists()

        self.pbar = tqdm(total=self.gen.max_frames - self.gen.frame_idx, desc="Processing", position=0, leave=True)

        self.run_prep_fn_list()


        # MAIN LOOP.
        while self.gen.frame_idx < self.gen.max_frames:
            frame_start = time.time()
            self.run_shoot_fn_list()

            self.pbar.update(self.gen.turbo_steps)
            if self.logging:
                duration = (time.time() - frame_start) * 1000
                self.logger.log(f"----------------------------- Frame {self.gen.frame_idx + 1} took {duration:.2f} ms")
        self.pbar.close()

        self.run_post_fn_list()
        if self.logging:
            self.logger.dump()
            total_duration = (time.time() - self.start_total_time) * 1000
            average_time_per_frame = (total_duration / self.gen.max_frames) if self.gen.max_frames!=0  else 0
            self.logger.log(f"Total time taken: {total_duration:.2f} ms")
            self.logger.log(f"Average time per frame: {average_time_per_frame:.2f} ms")
            self.logger.close_session()
        return self.gen

    def combined_pre_checks(self, settings_file: str = None, callback=None, *args, **kwargs):
        self.reset()
        self.setup_start = time.time()
        if callback is not None:
            self.datacallback = callback

        if self.logging and hasattr(self.logger, "start_session"):
            self.logger.start_session()
            self.start_total_time = time.time()
            duration = (self.start_total_time - self.script_start_time) * 1000
            self.logger.log(f"Script startup / model loading took {duration:.2f} ms")
        else:
            self.logging = False

        if settings_file:
            self.gen = DeforumGenerationObject.from_settings_file(settings_file)
        else:
            self.gen = DeforumGenerationObject(**kwargs)
        self.gen.update_from_kwargs(**kwargs)
        self.pre_setup()
        setup_end = time.time()
        duration = (setup_end - self.setup_start) * 1000
        if self.logging:
            self.logger.log(f"pre_setup took {duration:.2f} ms")

            setup_start = time.time()
        self.setup()

    def pre_setup(self):
        frame_warp_modes = ['2D', '3D']
        hybrid_motion_modes = ['Affine', 'Perspective', 'Optical Flow']

        # TODO WTF was this monstrosity?
        #self.gen.max_frames += 5

        # if self.gen.animation_mode in frame_warp_modes:
        #     # handle hybrid video generation
        if not self.gen.skip_hybrid_paths:
            if self.gen.hybrid_composite != 'None' or self.gen.hybrid_motion in hybrid_motion_modes:
                self.gen = hybrid_generation(self.gen)
                self.gen.hybrid_frame_path = os.path.join(self.gen.outdir, 'hybridframes')

        if not hasattr(self.gen, 'parseq_non_schedule_overrides'):
            self.gen.parseq_non_schedule_overrides = None

        if not hasattr(self.gen, 'enable_steps_scheduling'):
            self.gen.enable_steps_scheduling = False
        if not self.gen.enable_steps_scheduling:
            self.gen.steps_schedule = f"0: ({self.gen.steps})"

        # use parseq if manifest is provided
        #TODO Not passing controlnet_args yet
        self.parseq_adapter = ParseqAdapter(self.gen, self.gen, self.gen, None, self.gen)

        if int(self.gen.seed) == -1:
            self.gen.seed = secrets.randbelow(18446744073709551615)
        self.gen.keys = DeforumAnimKeys(self.gen, self.gen.seed) if not self.parseq_adapter.use_parseq else self.parseq_adapter.anim_keys
        self.gen.loopSchedulesAndData = LooperAnimKeys(self.gen, self.gen, self.gen.seed) if not self.parseq_adapter.use_parseq else self.parseq_adapter.looper_keys
        prompt_series = pd.Series([np.nan for a in range(self.gen.max_frames)])

        if self.gen.prompts is not None:
            if isinstance(self.gen.prompts, dict):
                self.gen.animation_prompts = self.gen.prompts

        for i, prompt in self.gen.animation_prompts.items():
            if str(i).isdigit():
                prompt_series[int(i)] = prompt
            else:
                prompt_series[int(numexpr.evaluate(i))] = prompt
        prompt_series = prompt_series.ffill().bfill()
        self.gen.prompt_series = prompt_series
        
        # TODO WTF was this monstrosity?
        #self.gen.max_frames -= 5

        # check for video inits
        self.gen.using_vid_init = self.gen.animation_mode == 'Video Input'

        # load depth model for 3D
        self.gen.predict_depths = self.gen.use_depth_warping or self.gen.save_depth_maps
        self.gen.predict_depths = self.gen.predict_depths or (
                self.gen.hybrid_composite and self.gen.hybrid_comp_mask_type in ['Depth', 'Video Depth'])
        if self.gen.predict_depths and (self.depth_model is None or self.loaded_depth_model != self.gen.depth_algorithm.lower()):
            # if self.opts is not None:
            #     self.keep_in_vram = self.opts.data.get("deforum_keep_3d_models_in_vram")
            # else:
            self.gen.keep_in_vram = True
            # device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else self.root.device)
            # TODO Set device in root in webui
            device = "cuda"
            self.depth_model = DepthModel(config.other_model_dir, device, self.gen.half_precision,
                                     keep_in_vram=self.gen.keep_in_vram,
                                     depth_algorithm=self.gen.depth_algorithm, Width=self.gen.width,
                                     Height=self.gen.height,
                                     midas_weight=self.gen.midas_weight)
            if 'adabins' in self.gen.depth_algorithm.lower():
                self.gen.use_adabins = True

                logger.info("Setting AdaBins usage")
            logger.info("[ Loaded Depth model ]")
            self.loaded_depth_model = self.gen.depth_algorithm.lower()
            # depth-based hybrid composite mask requires saved depth maps
            if self.gen.hybrid_composite != 'None' and self.gen.hybrid_comp_mask_type == 'Depth':
                self.gen.save_depth_maps = True
        else:
            if self.depth_model is not None:
                self.depth_model.to('cpu')
                del self.depth_model
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            self.depth_model = None
            self.gen.save_depth_maps = False

        load_raft = (self.gen.optical_flow_cadence == "RAFT" and int(self.gen.diffusion_cadence) > 0) or \
                    (self.gen.hybrid_motion == "Optical Flow" and self.gen.hybrid_flow_method == "RAFT") or \
                    (self.gen.optical_flow_redo_generation == "RAFT")
        if load_raft and self.raft_model is None:
            logger.info("[ Loading RAFT model ]")
            self.raft_model = RAFT()

        if self.gen.use_areas:

            try:
                self.gen.areas = interpolate_areas(self.gen.areas, self.gen.max_frames)
            except:
                self.gen.use_areas = False

    @deforumdoc
    def setup(self, *args, **kwargs) -> None:
        """
        Set up the list of functions to be executed during the main loop of the animation pipeline.

        This method populates the `shoot_fns` list with functions based on the configuration set in the `gen` object.
        Certain functions are added to the list based on the conditions provided by the attributes of the `gen` object.
        Additionally, post-processing functions can be added to the `post_fns` list.
        """
        hybrid_available = self.gen.hybrid_composite != 'None' or self.gen.hybrid_motion in ['Optical Flow', 'Affine', 'Perspective']

        self.shoot_fns.append(get_generation_params)

        self.gen.turbo_steps = self.gen.get('diffusion_cadence', 1)
        if self.gen.turbo_steps > 1:
            self.shoot_fns.append(generate_interpolated_frames)
        if self.gen.color_coherence == 'Video Input' and hybrid_available:
            self.shoot_fns.append(color_match_video_input)
        if self.gen.animation_mode in ['2D', '3D']:
            self.shoot_fns.append(anim_frame_warp_cls)

        if self.gen.hybrid_composite == 'Before Motion':
            self.shoot_fns.append(hybrid_composite_cls)

        if self.gen.hybrid_motion in ['Affine', 'Perspective']:
            self.shoot_fns.append(affine_persp_motion)

        if self.gen.hybrid_motion in ['Optical Flow']:
            self.shoot_fns.append(optical_flow_motion)

        if self.gen.hybrid_composite == 'Normal':
            self.shoot_fns.append(hybrid_composite_cls)

        if self.gen.color_coherence != 'None':
            self.shoot_fns.append(color_match_cls)

        self.shoot_fns.append(set_contrast_image)

        if self.gen.use_mask or self.gen.use_noise_mask:
            self.shoot_fns.append(handle_noise_mask)
        if self.gen.noise_type in ['perlin', 'uniform']:
            self.shoot_fns.append(add_noise_cls)

        if self.gen.optical_flow_redo_generation != 'None':
            self.shoot_fns.append(optical_flow_redo)

        if int(self.gen.diffusion_redo) > 0:
            self.shoot_fns.append(diffusion_redo)

        self.shoot_fns.append(main_generate_with_cls)

        if self.gen.hybrid_composite == 'After Generation':
            self.shoot_fns.append(post_hybrid_composite_cls)

        if self.gen.color_coherence != 'None':
            self.shoot_fns.append(post_color_match_with_cls)

        if self.gen.overlay_mask:
            self.shoot_fns.append(overlay_mask_cls)

        if hasattr(self.gen, "deforum_save_gen_info_as_srt"):
            if self.gen.deforum_save_gen_info_as_srt:
                self.shoot_fns.append(cls_subtitle_handler)

        self.shoot_fns.append(post_gen_cls)
        
        if hasattr(self.gen, 'enable_temporal_flow'):
            if self.gen.enable_temporal_flow and self.gen.turbo_steps < 2:
                self.shoot_fns.append(apply_temporal_flow_cls)
        if not self.gen.enable_ad_pass:
            if getattr(self.gen, "frame_interpolation_engine", None) and self.gen.frame_interpolation_engine != "None":
                if self.gen.max_frames > 3:
                    if self.gen.frame_interpolation_engine == "FILM":
                        self.post_fns.append(film_interpolate_cls)
                    elif 'rife' in self.gen.frame_interpolation_engine.lower():
                        self.post_fns.append(rife_interpolate_cls)
                    else:
                        raise ValueError(f"Unknown frame interpolation engine: {self.gen.frame_interpolation_engine}")

        if self.gen.max_frames > 1 and not self.gen.skip_video_creation:
            self.post_fns.append(save_video_cls)

        if self.gen.enable_ad_pass:
            self.post_fns.append(run_adiff_cls)


        os.makedirs(config.settings_path, exist_ok=True)
        settings_file_name = os.path.join(config.settings_path, f"{self.gen.timestring}_settings.txt")
        self.gen.save_as_json(settings_file_name)

        if self.gen.use_init and self.gen.init_image:

            if isinstance(self.gen.init_image, str):
                img = load_image(self.gen.init_image)
                img = np.array(img)
            elif isinstance(self.gen.init_image, PIL.Image.Image):
                img = np.array(self.gen.init_image)

            self.gen.prev_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.gen.opencv_image = self.gen.prev_img

        if self.gen.resume_from_timestring:
            def numeric_key(filename):
                # Extract the numeric part from the filename, assuming it follows the last underscore '_'
                parts = os.path.splitext(filename)[0].split('_')
                try:
                    return int(parts[-1])  # Convert the last part to integer
                except ValueError:
                    return 0  # Default to 0 if conversion fails

            resume_timestring = self.gen.resume_timestring
            if hasattr(self.gen, 'resume_from') and hasattr(self.gen, 'resume_from_frame'):
                if self.gen.resume_from_frame:
                    resume_from = self.gen.resume_from

                    parent_dir = os.path.dirname(self.gen.resume_path)

                    def strip_version_suffix(path):
                        # Regex to remove the version suffix from the path
                        return re.sub(r'_v\d+$', '', os.path.basename(path))
                    stripped = strip_version_suffix(self.gen.resume_path)
                    existing_versions = [d for d in os.listdir(parent_dir)
                                         if os.path.isdir(os.path.join(parent_dir, d)) and
                                         d.startswith(stripped + "_v")]
                    # Compute the next version number
                    if existing_versions:
                        # Extract version numbers and find the maximum
                        max_version = max(int(d.split('_v')[-1]) for d in existing_versions)
                        next_version = max_version + 1
                        new_outdir = self.gen.resume_path.replace(f"_v{max_version}", f"_v{next_version}")
                    else:
                        next_version = 1
                        new_outdir = os.path.join(parent_dir,
                                                  f"{os.path.basename(self.gen.resume_path)}_v{next_version}")

                    os.makedirs(new_outdir, exist_ok=True)
                    image_files = sorted(glob(os.path.join(self.gen.resume_path, '*.png')), key=numeric_key)
                    files_to_copy = image_files[:resume_from]
                    for file in files_to_copy:
                        shutil.copy(file, new_outdir)

                    self.gen.resume_path = new_outdir
                    if self.gen.max_frames <= resume_from:
                        self.gen.max_frames += 1

            image_files = sorted(glob(os.path.join(self.gen.resume_path, '*.png')), key=numeric_key)
            self.images = [Image.open(img_path) for img_path in image_files]
            self.image_paths = image_files

            batch_name, prev_frame, next_frame, prev_img, next_img = get_resume_vars(
                resume_path=self.gen.resume_path,
                timestring=self.gen.resume_timestring,
                cadence=self.gen.turbo_steps
            )
            batch_name = self.gen.resume_path.split('/')[-1]
            self.gen.timestring = resume_timestring
            self.gen.batch_name = batch_name
            self.gen.outdir = os.path.join(config.root_path, f"output/deforum/{batch_name}")

            if self.gen.turbo_steps > 1:
                self.gen.turbo_prev_image, self.gen.turbo_prev_frame_idx = prev_img, prev_frame
                self.gen.turbo_next_image, self.gen.turbo_next_frame_idx = next_img, next_frame
            start_frame = next_frame + 1
            self.gen.image = next_img
            self.gen.init_image = next_img
            self.gen.prev_img = prev_img
            self.gen.opencv_image = next_img
            self.gen.width, self.gen.height = prev_img.shape[1], prev_img.shape[0]
            self.gen.frame_idx = start_frame
            self.gen.use_init = True
        os.makedirs(self.gen.outdir, exist_ok=True)
        self.gen.image_paths = []


        if hasattr(self.gen, 'ip_adapter_image') and hasattr(self.generator, 'set_ip_adapter_image'):
            if self.gen.ip_adapter_image != "":
                ip_image = load_image(self.gen.ip_adapter_image)
                params = {
                    "weight": getattr(self.gen, "ip_adapter_strength", 1.0),
                    "start": getattr(self.gen, "ip_adapter_start", 0.0),
                    "end": getattr(self.gen, "ip_adapter_end", 1.0),
                }
                params = {k: v for k, v in params.items() if v is not None}
                self.generator.set_ip_adapter_image(ip_image, **params)
                logger.info(f"IP Adapter setup complete. Start: {params['start']}%, End: {params['end']}%, Weight: {params['weight']}")




    @deforumdoc
    def live_update_from_kwargs(self, **kwargs):
        """
        Updates the internal 'gen' object with the provided key-value arguments.

        :param kwargs dict: Key-value arguments to update the 'gen' object.

        Example usage:

        ```python
        animation_pipeline.live_update_from_kwargs(new_param1=value1, new_param2=value2)
        ```
        """
        try:
            if hasattr(self, 'gen'):
                self.gen.update_from_kwargs(**kwargs)
                self.gen.keys = DeforumAnimKeys(self.gen, self.gen.seed) if not self.parseq_adapter.use_parseq else self.parseq_adapter.anim_keys
                prompt_series = pd.Series([np.nan for a in range(self.gen.max_frames + 5)])

                if self.gen.prompts is not None:
                    if isinstance(self.gen.prompts, dict):
                        self.gen.animation_prompts = self.gen.prompts

                for i, prompt in self.gen.animation_prompts.items():
                    if str(i).isdigit():
                        prompt_series[int(i)] = prompt
                    else:
                        prompt_series[int(numexpr.evaluate(i))] = prompt
                prompt_series = prompt_series.ffill().bfill()
                self.gen.prompt_series = prompt_series
        except Exception as e:
            pass


    def log_function_lists(self):
        if self.logging:
            setup_end = time.time()
            duration = (setup_end - self.setup_start) * 1000
            self.logger.log(f"loop took {duration:.2f} ms")

            # Log names of functions in each list if they have functions
            if self.prep_fns:
                self.logger.log("Functions in prep_fns:", timestamped=False)
                for fn in self.prep_fns:
                    self.logger.log(fn.__name__, timestamped=False)

            if self.shoot_fns:
                self.logger.log("Functions in shoot_fns:", timestamped=False)
                for fn in self.shoot_fns:
                    self.logger.log(fn.__name__, timestamped=False)

            if self.post_fns:
                self.logger.log("Functions in post_fns:", timestamped=False)
                for fn in self.post_fns:
                    self.logger.log(fn.__name__, timestamped=False)

            self.logger.log(str(self.gen.to_dict()), timestamped=False)

    def run_prep_fn_list(self):
        # PREP LOOP
        for fn in self.prep_fns:
            start_time = time.time()
            fn(self)
            if self.logging:
                end_time = time.time()
                duration = (end_time - start_time) * 1000
                self.logger.log(f"{fn.__name__} took {duration:.2f} ms")

    def run_shoot_fn_list(self):
        for fn in self.shoot_fns:
            if not self.interrupt:
                start_time = time.time()
                with torch.inference_mode():
                    with torch.inference_mode():
                        fn(self)
                if self.logging:
                    end_time = time.time()
                    duration = (end_time - start_time) * 1000
                    self.logger.log(f"{fn.__name__} took {duration:.2f} ms")
            else:
                self.gen.frame_idx = self.gen.max_frames
    def run_post_fn_list(self):

        # if self.gen.enable_ad_pass:
        self.cleanup()

        # POST LOOP
        for fn in self.post_fns:
            start_time = time.time()
            if not self.interrupt:
                fn(self)
                if self.logging:
                    duration = (time.time() - start_time) * 1000
                    self.logger.log(f"{fn.__name__} took {duration:.2f} ms")
            else:
                self.gen.frame_idx = self.gen.max_frames

    @deforumdoc
    def reset(self, *args, **kwargs) -> None:
        """
        Cleans up the resources used by the pipeline by freeing GPU memory and deleting attributes.

        :return: None.
        :rtype: None

        Example usage:

        ```python
        animation_pipeline.cleanup()
        ```
        """
        self.prep_fns.clear()
        self.shoot_fns.clear()
        self.post_fns.clear()
        self.images.clear()
        if hasattr(self, 'gen'):
            del self.gen
            self.gen = None
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        gc.collect()

    def datacallback(self, data) -> None:
        return None


    def generate(self):
        """
        Generates an image or animation using the given prompts, settings, and generator.

        This method sets up the necessary arguments, handles conditional configurations, and then
        uses the provided generator to produce the output.

        Returns:
            processed (Image): The generated image or animation frame.
        """
        prompt, negative_prompt = split_weighted_subprompts(self.gen.prompt, self.gen.frame_idx, self.gen.max_frames)
        next_prompt, blend_value = get_next_prompt_and_blend(self.gen.frame_idx, self.gen.prompt_series)
        init_image = None
        cnet_image = None
        # if not self.gen.use_init and self.gen.strength < 1.0 and self.gen.strength_0_no_init:
        #     self.gen.strength = 1.0
        if hasattr(self.gen, "sampler_name"):
            if isinstance(self.generator, ComfyDeforumGenerator):
                try:
                    from comfy.samplers import SAMPLER_NAMES
                    if self.gen.sampler_name not in SAMPLER_NAMES:
                        sampler_name = auto_to_comfy[self.gen.sampler_name]["sampler"]
                        scheduler = auto_to_comfy[self.gen.sampler_name]["scheduler"]
                        self.gen.sampler_name = sampler_name
                        self.gen.scheduler = scheduler
                except Exception:
                    logger.info("No Comfy available when setting scheduler name")
        if self.gen.scheduled_sampler_name is not None and self.gen.enable_sampler_scheduling:
            if self.gen.scheduled_sampler_name in auto_to_comfy.keys():
                self.gen.sampler_name = auto_to_comfy[self.gen.sampler_name]["sampler"]
                self.gen.scheduler = auto_to_comfy[self.gen.sampler_name]["scheduler"]

        # logger.info(f"GENERATE'S SAMPLER NAME: {self.gen.sampler_name}, {self.gen.scheduler}")
        # if self.gen.prev_img is not None:
        #     # TODO: cleanup init_sample remains later
        #     init_image = cv2.cvtColor(self.gen.prev_img, cv2.COLOR_BGR2RGB)
            # init_image = img
        if self.gen.frame_idx > 0:
            self.gen.use_init = False
        # if self.gen.use_init and self.gen.init_image:
        #     if not isinstance(self.gen.init_image, PIL.Image.Image):
        #         self.gen.init_image = Image.open(self.gen.init_image)
        #     init_image = np.array(self.gen.init_image).astype(np.uint8)
        if self.gen.prev_img is not None:
            self.gen.init_sample = Image.fromarray(cv2.cvtColor(self.gen.prev_img, cv2.COLOR_BGR2RGB))
        gen_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": self.gen.steps,
            "seed": self.gen.seed,
            "scale": self.gen.scale,
            "strength": self.gen.strength,
            "init_image": self.gen.init_sample,
            "width": self.gen.width,
            "height": self.gen.height,
            "cnet_image": cnet_image,
            "next_prompt": next_prompt,
            "prompt_blend": blend_value,
            "scheduler": self.gen.scheduler,
            "sampler_name": self.gen.sampler_name,
            "reset_noise": False if self.gen.strength < 1.0 else True,
        }
        if self.gen.frame_idx == 0 and not self.gen.use_init:
            gen_args["reset_noise"] = True
        if hasattr(self.gen, "style"):
            if self.gen.style != "(No Style)" and self.gen.style in STYLE_NAMES:
                gen_args["prompt"], gen_args["negative_prompt"] = apply_style(self.gen.style, gen_args["prompt"],
                                                                              gen_args["negative_prompt"])

        if self.gen.use_areas:
            gen_args["areas"] = self.gen.areas[self.gen.frame_idx]
            gen_args["use_areas"] = True
            gen_args["prompt"] = None

        if self.gen.enable_subseed_scheduling:
            gen_args["subseed"] = self.gen.subseed
            gen_args["subseed_strength"] = self.gen.subseed_strength
            gen_args["seed_resize_from_h"] = self.gen.seed_resize_from_h
            gen_args["seed_resize_from_w"] = self.gen.seed_resize_from_w
        if hasattr(self.gen, 'animation_prompts_positive'):
            gen_args["prompt"] = gen_args["prompt"] + self.gen.animation_prompts_positive
            if next_prompt:
                gen_args['next_prompt'] = next_prompt + self.gen.animation_prompts_positive
        if hasattr(self.gen, 'animation_prompts_negative'):
            gen_args["negative_prompt"] = gen_args["negative_prompt"] + self.gen.animation_prompts_negative
        def calculate_blend_factor(frame_idx, cycle_length=100):
            """Calculate the blending factor for the frame index within a cycle of length `cycle_length`.
            The factor linearly increases from 0 to 1 and then resets."""
            phase = frame_idx % cycle_length
            return phase / (cycle_length - 1)
        if not self.gen.dry_run:

            processed = self.generator(**gen_args)

        else:
            # Get the path to the default image or use the previous image
            if self.gen.prev_img is None:
                default_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ui',
                                                  'deforum.png')
                default_image = Image.open(default_image_path).resize((self.gen.width, self.gen.height),
                                                                      resample=Image.Resampling.LANCZOS)
            else:
                default_image = Image.fromarray(cv2.cvtColor(self.gen.prev_img, cv2.COLOR_BGR2RGB))

            # Calculate blend factor
            blend_factor = 0.01

            # Determine the original image to blend with
            if self.gen.first_frame is None:
                self.gen.first_frame = default_image

            # Blend images
            processed = Image.blend(default_image, self.gen.first_frame, blend_factor)

            # Overlay frame number on the image
            draw = ImageDraw.Draw(processed)
            text_position = (120, 120)
            text="Diffused frame: " + str(self.gen.frame_idx) + "/" + str(self.gen.max_frames-1)
            font=ImageFont.load_default(size=50)
            bbox = draw.textbbox(xy=text_position, text=text, font=font)

            background_position = (bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5)
            draw.rectangle(background_position, fill="black")
            draw.text(xy=text_position, text=text, font=font, fill="white")

            # Update first_frame at the end of each cycle
            if (self.gen.frame_idx + 1) % 200 == 0:  # Reset first frame after each full cycle
                self.gen.first_frame = processed

        # These values are used in generate_interpolated_frames() to known between which frames
        # to interpolate on the next cycle around the MAIN LOOP.
        # Note that self.gen.next_frame_to_diffuse may be greater than max_frames – this is OK,
        # as it will be clamped in generate_interpolated_frames().
        self.gen.last_diffused_frame = self.gen.frame_idx
        self.gen.next_frame_to_diffuse = self.gen.frame_idx + self.gen.turbo_steps

        return processed

    def cleanup(self):
        # Iterate over all attributes of the class instance
        if hasattr(self.generator, 'cleanup'):
            self.generator.cleanup()

        for attr_name in dir(self):
            attr = getattr(self, attr_name)

            # Check if attribute is a tensor
            if isinstance(attr, torch.Tensor):
                attr.cpu()  # Move tensor to CPU
                delattr(self, attr_name)  # Delete attribute
            # Check if attribute is an nn.Module
            elif isinstance(attr, nn.Module):
                attr.to("cpu")  # Move module to CPU
                delattr(self, attr_name)  # Delete attribute
            # Check if attribute has 'to' or 'cpu' callable method
            else:
                to_method = getattr(attr, "to", None)
                cpu_method = getattr(attr, "cpu", None)

                if callable(to_method):
                    attr.to("cpu")
                    delattr(self, attr_name)  # Delete attribute
                elif callable(cpu_method):
                    attr.cpu()
                    delattr(self, attr_name)  # Delete attribute
        self.depth_model = None
        self.raft_model = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


def interpolate_areas(areas, max_frames):
    # Special case for max_frames = 1
    if max_frames == 1:
        return [areas[0]["0"]]

    # Parse the keyframes and their values
    keyframes = sorted([int(k) for area in areas for k in area])

    # If there are more keyframes than max_frames, truncate the list of keyframes
    keyframes = keyframes[:max_frames]

    # Initialize the result list with empty dicts for each frame
    result = [{} for _ in range(max_frames + 1)]

    # For each pair of consecutive keyframes
    for i in range(len(keyframes) - 1):
        start_frame = keyframes[i]
        end_frame = keyframes[i + 1]
        start_prompts = areas[i][str(start_frame)]
        end_prompts = areas[i + 1][str(end_frame)]

        # Create a mapping of prompts to their values for easier lookup
        start_mapping = {prompt['prompt']: prompt for prompt in start_prompts}
        end_mapping = {prompt['prompt']: prompt for prompt in end_prompts}

        # Combine the set of prompts from both start and end keyframes
        all_prompts = set(start_mapping.keys()) | set(end_mapping.keys())

        # For each frame between the start and end keyframes
        end_range = min(end_frame + 1, max_frames + 1)
        for frame in range(start_frame, end_range):
            result[frame] = []
            for prompt in all_prompts:
                start_values = start_mapping.get(prompt, None)
                end_values = end_mapping.get(prompt, None)

                # If the prompt is only in the start or end keyframe, use its values
                if start_values is None:
                    result[frame].append(end_values)
                elif end_values is None:
                    result[frame].append(start_values)
                else:
                    # Otherwise, interpolate the values
                    t = (frame - start_frame) / (end_frame - start_frame)
                    x = start_values['x'] + t * (end_values['x'] - start_values['x'])
                    y = start_values['y'] + t * (end_values['y'] - start_values['y'])
                    w = start_values['w'] + t * (end_values['w'] - start_values['w'])
                    h = start_values['h'] + t * (end_values['h'] - start_values['h'])
                    s = start_values['s'] + t * (end_values['s'] - start_values['s'])
                    result[frame].append({'prompt': prompt, 'x': x, 'y': y, 'w': w, 'h': h, 's': s})

    # Fill in any remaining frames with the values from the last keyframe
    for frame in range(keyframes[-1], max_frames):
        result[frame] = areas[-1][str(keyframes[-1])]

    return result[:max_frames]


def generate_blend_values(distance_to_next_prompt, blend_type="linear"):
    if blend_type == "linear":
        return [i / distance_to_next_prompt for i in range(distance_to_next_prompt + 1)]
    elif blend_type == "exponential":
        base = 2
        return [1 / (1 + math.exp(-8 * (i / distance_to_next_prompt - 0.5))) for i in
                range(distance_to_next_prompt + 1)]
    else:
        raise ValueError(f"Unknown blend type: {blend_type}")


def get_next_prompt_and_blend(current_index, prompt_series, blend_type="exponential"):
    # Find where the current prompt ends
    next_prompt_start = current_index + 1
    while next_prompt_start < len(prompt_series) and prompt_series.iloc[next_prompt_start] == \
            prompt_series.iloc[
                current_index]:
        next_prompt_start += 1

    if next_prompt_start >= len(prompt_series):
        return "", 1.0
        # raise ValueError("Already at the last prompt, no next prompt available.")

    # Calculate blend value
    distance_to_next = next_prompt_start - current_index
    blend_values = generate_blend_values(distance_to_next, blend_type)
    blend_value = blend_values[1]  # Blend value for the next frame after the current index

    return prompt_series.iloc[next_prompt_start], blend_value




