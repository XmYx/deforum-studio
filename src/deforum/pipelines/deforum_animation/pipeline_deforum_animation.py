import gc
import json
import math
import os
import secrets
import time
from typing import Callable, Optional

import PIL.Image
import cv2
import numexpr
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from .animation_helpers import (
    anim_frame_warp_cls,
    hybrid_composite_cls,
    affine_persp_motion,
    optical_flow_motion,
    color_match_cls,
    set_contrast_image,
    handle_noise_mask,
    add_noise_cls,
    get_generation_params,
    optical_flow_redo,
    diffusion_redo,
    main_generate_with_cls,
    post_hybrid_composite_cls,
    post_color_match_with_cls,
    overlay_mask_cls,
    post_gen_cls,
    make_cadence_frames,
    color_match_video_input,
    film_interpolate_cls,
    save_video_cls,
    DeforumAnimKeys,
    LooperAnimKeys,
    generate_interpolated_frames
)

from .parseq_adapter import ParseqAdapter

from .animation_params import auto_to_comfy
from ..deforum_pipeline import DeforumBase
from ...models import DepthModel, RAFT
from ...pipeline_utils import DeforumGenerationObject, pairwise_repl, isJson
from ...utils.constants import root_path, other_model_dir
from ...utils.deforum_hybrid_animation import hybrid_generation
from ...utils.deforum_logger_util import Logger
from ...utils.image_utils import load_image_with_mask, prepare_mask, check_mask_for_errors, load_image
from ...utils.sdxl_styles import STYLE_NAMES, apply_style
from ...utils.string_utils import split_weighted_subprompts, check_is_number
from ...utils.video_frame_utils import get_frame_name


class DeforumAnimationPipeline(DeforumBase):
    """
    Animation pipeline for Deforum.

    Provides a mechanism to run an animation generation process using the provided generator.
    Allows for pre-processing, main loop, and post-processing steps.
    Uses a logger to record the metrics and timings of each step in the pipeline.
    """
    script_start_time = time.time()
    def __init__(self, generator: Callable, logger: Optional[Callable] = None):
        """
        Initialize the DeforumAnimationPipeline.

        Args:
            generator (Callable): The generator function for producing animations.
            logger (Optional[Callable], optional): Optional logger function. Defaults to None.
        """
        super().__init__()

        self.generator = generator

        if logger is None:
            self.logger = Logger(root_path)
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

    def __call__(self, settings_file: str = None, callback=None, *args, **kwargs) -> DeforumGenerationObject:
        """
        Execute the animation pipeline.

        Args:
            settings_file (str, optional): Path to the settings file. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            DeforumGenerationObject: The generated object after the pipeline execution.
        """
        self.interrupt = False

        self.combined_pre_checks(settings_file, callback, *args, **kwargs)

        self.log_function_lists()

        self.pbar = tqdm(total=self.gen.max_frames, desc="Processing", position=0, leave=True)

        self.run_prep_fn_list()


        while self.gen.frame_idx + 1 <= self.gen.max_frames:
            # MAIN LOOP
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
            average_time_per_frame = total_duration / self.gen.max_frames
            self.logger.log(f"Total time taken: {total_duration:.2f} ms")
            self.logger.log(f"Average time per frame: {average_time_per_frame:.2f} ms")
            self.logger.close_session()
        return self.gen

    def combined_pre_checks(self, settings_file: str = None, callback=None, *args, **kwargs):
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

        self.gen.max_frames += 1

        # if self.gen.animation_mode in frame_warp_modes:
        #     # handle hybrid video generation
        if not self.gen.skip_hybrid_paths:
            if self.gen.hybrid_composite != 'None' or self.gen.hybrid_motion in hybrid_motion_modes:
                _, _, self.gen.inputfiles = hybrid_generation(self.gen, self.gen, self.gen)
                self.gen.hybrid_frame_path = os.path.join(self.gen.outdir, 'hybridframes')


        # use parseq if manifest is provided
        self.parseq_adapter = ParseqAdapter(self.gen, self.gen, self.gen, self.gen, self.gen)

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
        self.gen.max_frames -= 1

        # check for video inits
        self.gen.using_vid_init = self.gen.animation_mode == 'Video Input'

        # load depth model for 3D
        self.gen.predict_depths = self.gen.use_depth_warping or self.gen.save_depth_maps
        self.gen.predict_depths = self.gen.predict_depths or (
                self.gen.hybrid_composite and self.gen.hybrid_comp_mask_type in ['Depth', 'Video Depth'])
        if self.gen.predict_depths:
            # if self.opts is not None:
            #     self.keep_in_vram = self.opts.data.get("deforum_keep_3d_models_in_vram")
            # else:
            self.gen.keep_in_vram = True
            # device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else self.root.device)
            # TODO Set device in root in webui
            device = "cuda"
            self.depth_model = DepthModel(other_model_dir, device, self.gen.half_precision,
                                     keep_in_vram=self.gen.keep_in_vram,
                                     depth_algorithm=self.gen.depth_algorithm, Width=self.gen.width,
                                     Height=self.gen.height,
                                     midas_weight=self.gen.midas_weight)
            if 'adabins' in self.gen.depth_algorithm.lower():
                self.gen.use_adabins = True
                print("Setting AdaBins usage")
            print(f"[ Loaded Depth model ]")
            # depth-based hybrid composite mask requires saved depth maps
            if self.gen.hybrid_composite != 'None' and self.gen.hybrid_comp_mask_type == 'Depth':
                self.gen.save_depth_maps = True
        else:
            self.depth_model = None
            self.gen.save_depth_maps = False

        self.raft_model = None
        load_raft = (self.gen.optical_flow_cadence == "RAFT" and int(self.gen.diffusion_cadence) > 0) or \
                    (self.gen.hybrid_motion == "Optical Flow" and self.gen.hybrid_flow_method == "RAFT") or \
                    (self.gen.optical_flow_redo_generation == "RAFT")
        if load_raft:
            print("[ Loading RAFT model ]")
            self.raft_model = RAFT()

        if self.gen.use_areas:

            try:
                self.gen.areas = interpolate_areas(self.gen.areas, self.gen.max_frames)
            except:
                self.gen.use_areas = False

    def setup(self, *args, **kwargs) -> None:
        """
        Set up the list of functions to be executed during the main loop of the animation pipeline.

        This method populates the `shoot_fns` list with functions based on the configuration set in the `gen` object.
        Certain functions are added to the list based on the conditions provided by the attributes of the `gen` object.
        Additionally, post-processing functions can be added to the `post_fns` list.
        """
        self.reset()

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

        self.shoot_fns.append(post_gen_cls)

        if self.gen.max_frames > 3:
            if self.gen.frame_interpolation_engine == "FILM":
                self.post_fns.append(film_interpolate_cls)
        if self.gen.max_frames > 1 and not self.gen.skip_video_creation:
            self.post_fns.append(save_video_cls)
        os.makedirs("deforum_configs", exist_ok=True)
        settings_file_name = os.path.join("deforum_configs", f"{self.gen.timestring}_settings.txt")
        self.gen.save_as_json(settings_file_name)

        if self.gen.use_init and self.gen.init_image:

            if isinstance(self.gen.init_image, str):
                img = load_image(self.gen.init_image)
                img = np.array(img)
            elif isinstance(self.gen.init_image, PIL.Image.Image):
                img = np.array(self.gen.init_image)

            self.gen.prev_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.gen.opencv_image = self.gen.prev_img

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


    def reset(self, *args, **kwargs) -> None:
        self.prep_fns.clear()
        self.shoot_fns.clear()
        self.post_fns.clear()
        self.images.clear()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        gc.collect()

    def datacallback(self, data):
        pass


    def generate_(self):
        assert self.gen.prompt is not None
        prompt, negative_prompt = split_weighted_subprompts(self.gen.prompt, self.gen.frame_idx, self.gen.max_frames)

        next_prompt, blend_value = get_next_prompt_and_blend(self.gen.frame_idx, self.gen.prompt_series)


        if hasattr(self.gen, "sampler_name"):
            from comfy.samplers import SAMPLER_NAMES


            if self.gen.sampler_name not in SAMPLER_NAMES:
                sampler_name = auto_to_comfy[self.gen.sampler_name]["sampler"]
                scheduler = auto_to_comfy[self.gen.sampler_name]["scheduler"]
                self.gen.sampler_name = sampler_name
                self.gen.scheduler = scheduler

        if self.gen.scheduled_sampler_name is not None:
            if self.gen.scheduled_sampler_name in auto_to_comfy.keys():
                sampler_name = auto_to_comfy[self.gen.sampler_name]["sampler"]
                scheduler = auto_to_comfy[self.gen.sampler_name]["scheduler"]
                self.gen.sampler_name = sampler_name
                self.gen.scheduler = scheduler

        img = None
        if self.gen.opencv_image is not None:
            if not isinstance(img, PIL.Image.Image):
                img = Image.fromarray(cv2.cvtColor(self.gen.opencv_image.astype(np.uint8), cv2.COLOR_BGR2RGB))

        # if self.gen.use_init and self.gen.init_image:
        #     img = self.gen.init_image


        self.gen.strength = 1.0 if img is None else self.gen.strength

        gen_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": self.gen.steps,
            "seed": self.gen.seed,
            "scale": self.gen.scale,
            "strength": self.gen.strength,
            "init_image": img,
            "width": self.gen.width,
            "height": self.gen.height,
            "cnet_image": None,
            "next_prompt": next_prompt,
            "prompt_blend": blend_value,
            "scheduler": self.gen.scheduler,
            "sampler_name": self.gen.sampler_name,
            "reset_noise": False if self.gen.strength < 1.0 else True
        }
        if self.gen.frame_idx == 0 and not self.gen.use_init:
            gen_args["reset_noise"] = True

        if hasattr(self.gen, "style"):
            if self.gen.style is not "(No Style)" and self.gen.style in STYLE_NAMES:
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

        processed = self.generator(**gen_args)
        torch.cuda.empty_cache()

        return processed

    def generate(self):
        """
        Generates an image or animation using the given prompts, settings, and generator.

        This method sets up the necessary arguments, handles conditional configurations, and then
        uses the provided generator to produce the output.

        Returns:
            processed (Image): The generated image or animation frame.
        """
        assert self.gen.prompt is not None
        prompt, negative_prompt = split_weighted_subprompts(self.gen.prompt, self.gen.frame_idx, self.gen.max_frames)
        next_prompt, blend_value = get_next_prompt_and_blend(self.gen.frame_idx, self.gen.prompt_series)
        # next_prompt = ""
        if not self.gen.use_init and self.gen.strength < 1.0 and self.gen.strength_0_no_init:
            self.gen.strength = 1.0
        processed = None
        mask_image = None
        init_image = None
        image_init0 = None

        if hasattr(self.gen, "sampler_name"):
            from comfy.samplers import SAMPLER_NAMES
            if self.gen.sampler_name not in SAMPLER_NAMES:
                sampler_name = auto_to_comfy[self.gen.sampler_name]["sampler"]
                scheduler = auto_to_comfy[self.gen.sampler_name]["scheduler"]
                self.gen.sampler_name = sampler_name
                self.gen.scheduler = scheduler

        if self.gen.scheduled_sampler_name is not None and self.gen.enable_sampler_scheduling:
            if self.gen.scheduled_sampler_name in auto_to_comfy.keys():
                self.gen.sampler_name = auto_to_comfy[self.gen.sampler_name]["sampler"]
                self.gen.scheduler = auto_to_comfy[self.gen.sampler_name]["scheduler"]

        print("GENERATE'S SAMPLER NAME", self.gen.sampler_name, self.gen.scheduler)

        if self.gen.use_looper and self.gen.animation_mode in ['2D', '3D']:
            self.gen.strength = self.gen.imageStrength
            tweeningFrames = self.gen.tweeningFrameSchedule
            blendFactor = .07
            colorCorrectionFactor = self.gen.colorCorrectionFactor
            jsonImages = json.loads(self.gen.imagesToKeyframe)
            # find which image to show
            parsedImages = {}
            frameToChoose = 0
            max_f = self.gen.max_frames - 1

            for key, value in jsonImages.items():
                if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
                    parsedImages[key] = value
                else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                    parsedImages[int(numexpr.evaluate(key))] = value

            framesToImageSwapOn = list(map(int, list(parsedImages.keys())))

            for swappingFrame in framesToImageSwapOn[1:]:
                frameToChoose += (self.gen.frame_idx >= int(swappingFrame))

            # find which frame to do our swapping on for tweening
            skipFrame = 25
            for fs, fe in pairwise_repl(framesToImageSwapOn):
                if fs <= self.gen.frame_idx <= fe:
                    skipFrame = fe - fs
            if skipFrame > 0:
                # print("frame % skipFrame", frame % skipFrame)

                if self.gen.frame_idx % skipFrame <= tweeningFrames:  # number of tweening frames
                    blendFactor = self.gen.blendFactorMax - self.gen.blendFactorSlope * math.cos(
                        (self.gen.frame_idx % tweeningFrames) / (tweeningFrames / 2))
            else:
                print("LOOPER ERROR, AVOIDING DIVISION BY 0")
            init_image2, _ = load_image_with_mask(list(jsonImages.values())[frameToChoose],
                                      shape=(self.gen.width, self.gen.height),
                                      use_alpha_as_mask=self.gen.use_alpha_as_mask)
            image_init0 = list(jsonImages.values())[0]
            # print(" TYPE", type(image_init0))


        else:  # they passed in a single init image
            image_init0 = self.gen.init_image

        available_samplers = {
            'euler a': 'Euler a',
            'euler': 'Euler',
            'lms': 'LMS',
            'heun': 'Heun',
            'dpm2': 'DPM2',
            'dpm2 a': 'DPM2 a',
            'dpm++ 2s a': 'DPM++ 2S a',
            'dpm++ 2m': 'DPM++ 2M',
            'dpm++ sde': 'DPM++ SDE',
            'dpm fast': 'DPM fast',
            'dpm adaptive': 'DPM adaptive',
            'lms karras': 'LMS Karras',
            'dpm2 karras': 'DPM2 Karras',
            'dpm2 a karras': 'DPM2 a Karras',
            'dpm++ 2s a karras': 'DPM++ 2S a Karras',
            'dpm++ 2m karras': 'DPM++ 2M Karras',
            'dpm++ sde karras': 'DPM++ SDE Karras'
        }


            # else:
            #     raise RuntimeError(
            #         f"Sampler name '{sampler_name}' is invalid. Please check the available sampler list in the 'Run' tab")

        # if self.gen.checkpoint is not None:
        #    info = sd_models.get_closet_checkpoint_match(self.gen.checkpoint)
        #    if info is None:
        #        raise RuntimeError(f"Unknown checkpoint: {self.gen.checkpoint}")
        #    sd_models.reload_model_weights(info=info)

        if self.gen.prev_img is not None:
            # TODO: cleanup init_sample remains later
            img = cv2.cvtColor(self.gen.prev_img, cv2.COLOR_BGR2RGB)



            init_image = img
            image_init0 = img
            if self.gen.use_looper and isJson(self.gen.imagesToKeyframe) and self.gen.animation_mode in ['2D', '3D']:
                init_image = Image.blend(init_image, init_image2, blendFactor)
                correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
                color_corrections = [correction_colors]

        # this is the first pass
        elif (self.gen.use_looper and self.gen.animation_mode in ['2D', '3D']) or (
                self.gen.use_init and ((self.gen.init_image is not None and self.gen.init_image != ''))):
            init_image, mask_image = load_image_with_mask(image_init0,  # initial init image
                                              shape=(self.gen.width, self.gen.height),
                                              use_alpha_as_mask=self.gen.use_alpha_as_mask)

        else:

            # if self.gen.animation_mode != 'Interpolation':
            #    print(f"Not using an init image (doing pure txt2img)")
            """p_txt = StableDiffusionProcessingTxt2Img( 
                sd_model=sd_model,
                outpath_samples=self.gen.tmp_deforum_run_duplicated_folder,
                outpath_grids=self.gen.tmp_deforum_run_duplicated_folder,
                prompt=p.prompt,
                styles=p.styles,
                negative_prompt=p.negative_prompt,
                seed=p.seed,
                subseed=p.subseed,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                sampler_name=p.sampler_name,
                batch_size=p.batch_size,
                n_iter=p.n_iter,
                steps=p.steps,
                cfg_scale=p.cfg_scale,
                width=p.width,
                height=p.height,
                restore_faces=p.restore_faces,
                tiling=p.tiling,
                enable_hr=None,
                denoising_strength=None,
            )"""

            # print_combined_table(args, anim_args, p_txt, keys, frame)  # print dynamic table to cli

            # if is_controlnet_enabled(controlnet_args):
            #    process_with_controlnet(p_txt, args, anim_args, loop_args, controlnet_args, root, is_img2img=False,
            #                            self.gen.frame_idx=frame)

            # processed = self.generate_txt2img(prompt, next_prompt, blend_value, negative_prompt, args, anim_args, root, self.gen.frame_idx,
            #                                init_image)

            self.gen.strength = 1.0 if init_image is None else self.gen.strength

            cnet_image = None
            input_file = os.path.join(self.gen.outdir, 'inputframes',
                                      get_frame_name(self.gen.video_init_path) + f"{self.gen.frame_idx:09}.jpg")

            # if os.path.isfile(input_file):
            #     input_frame = Image.open(input_file)
            #     cnet_image = get_canny_image(input_frame)
            #     cnet_image = ImageOps.invert(cnet_image)

            if prompt == "!reset!":
                self.gen.init_image = None
                self.gen.strength = 1.0
                prompt = next_prompt

            if negative_prompt == "":
                negative_prompt = self.gen.animation_prompts_negative

            gen_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": self.gen.steps,
                "seed": self.gen.seed,
                "scale": self.gen.scale,
                "strength": self.gen.strength,
                "init_image": init_image,
                "width": self.gen.width,
                "height": self.gen.height,
                "cnet_image": cnet_image,
                "next_prompt": next_prompt,
                "prompt_blend": blend_value,
                "scheduler":self.gen.scheduler,
                "sampler_name":self.gen.sampler_name,
                "reset_noise":False if self.gen.strength < 1.0 else True
            }
            if self.gen.frame_idx == 0:
                gen_args["reset_noise"] = True
            if hasattr(self.gen, "style"):
                if self.gen.style is not "(No Style)" and self.gen.style in STYLE_NAMES:
                    gen_args["prompt"], gen_args["negative_prompt"] = apply_style(self.gen.style, gen_args["prompt"], gen_args["negative_prompt"])


            if self.gen.use_areas:
                gen_args["areas"] = self.gen.areas[self.gen.frame_idx]
                gen_args["use_areas"] = True
                gen_args["prompt"] = None




            if self.gen.enable_subseed_scheduling:
                gen_args["subseed"] = self.gen.subseed
                gen_args["subseed_strength"] = self.gen.subseed_strength
                gen_args["seed_resize_from_h"] = self.gen.seed_resize_from_h
                gen_args["seed_resize_from_w"] = self.gen.seed_resize_from_w

            processed = self.generator(**gen_args)
            torch.cuda.empty_cache()

        if processed is None:
            # Mask functions
            if self.gen.use_mask:
                mask_image = self.gen.mask_image
                mask = prepare_mask(self.gen.mask_file if mask_image is None else mask_image,
                                    (self.gen.width, self.gen.H),
                                    self.gen.mask_contrast_adjust,
                                    self.gen.mask_brightness_adjust)
                inpainting_mask_invert = self.gen.invert_mask
                inpainting_fill = self.gen.fill
                inpaint_full_res = self.gen.full_res_mask
                inpaint_full_res_padding = self.gen.full_res_mask_padding
                # prevent loaded mask from throwing errors in Image operations if completely black and crop and resize in webui pipeline
                # doing this after contrast and brightness adjustments to ensure that mask is not passed as black or blank
                mask = check_mask_for_errors(mask, self.gen.invert_mask)
                self.gen.noise_mask = mask

            else:
                mask = None

            assert not ((mask is not None and self.gen.use_mask and self.gen.overlay_mask) and (
                    self.gen.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

            image_mask = mask
            image_cfg_scale = self.gen.pix2pix_img_cfg_scale

            # print_combined_table(args, anim_args, p, keys, frame)  # print dynamic table to cli

            # if is_controlnet_enabled(controlnet_args):
            #    process_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, is_img2img=True,
            #                            self.gen.frame_idx=frame)
            self.gen.strength = 1.0 if init_image is None else self.gen.strength

            cnet_image = None
            input_file = os.path.join(self.gen.outdir, 'inputframes',
                                      get_frame_name(self.gen.video_init_path) + f"{self.gen.frame_idx:09}.jpg")

            # if os.path.isfile(input_file):
            #     input_frame = Image.open(input_file)
            #     cnet_image = get_canny_image(input_frame)
            #     cnet_image = ImageOps.invert(cnet_image)

            if prompt == "!reset!":
                init_image = None
                self.gen.strength = 1.0
                prompt = next_prompt

            if negative_prompt == "":
                negative_prompt = self.gen.animation_prompts_negative

            gen_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": self.gen.steps,
                "seed": self.gen.seed,
                "scale": self.gen.scale,
                "strength": self.gen.strength,
                "init_image": init_image,
                "width": self.gen.width,
                "height": self.gen.height,
                "cnet_image": cnet_image,
                "next_prompt": next_prompt,
                "prompt_blend": blend_value,
                "scheduler": self.gen.scheduler,
                "sampler_name": self.gen.sampler_name,
                "reset_noise": False if self.gen.strength < 1.0 else True
            }

            if self.gen.use_areas:

                gen_args["areas"] =  self.gen.areas[self.gen.frame_idx]
                gen_args["use_areas"] = True
                gen_args["prompt"] = None
                #print(f"DEFORUM GEN ARGS: [{gen_args}] ")

            if self.gen.enable_subseed_scheduling:
                gen_args["subseed"] = self.gen.subseed
                gen_args["subseed_strength"] = self.gen.subseed_strength
                gen_args["seed_resize_from_h"] = self.gen.seed_resize_from_h
                gen_args["seed_resize_from_w"] = self.gen.seed_resize_from_w


            processed = self.generator(**gen_args)

        if self.gen.first_frame is None:
            self.gen.first_frame = processed

        return processed

    def cleanup(self):
        # Iterate over all attributes of the class instance
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
