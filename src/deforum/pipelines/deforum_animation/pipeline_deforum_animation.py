import gc
import json
import math
import os
import secrets
import time
from typing import Callable, Optional

import numexpr
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from PIL import Image

from .animation_helpers import (anim_frame_warp_cls,
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
                                save_video_cls, DeformAnimKeys, LooperAnimKeys)

from .animation_params import auto_to_comfy

from ..deforum_pipeline import DeforumBase

from ...models import DepthModel, RAFT
from ...pipeline_utils import DeforumGenerationObject, pairwise_repl, isJson
from ...utils.constants import root_path, other_model_dir
from ...utils.deforum_hybrid_animation import hybrid_generation
from ...utils.deforum_logger_util import Logger
from ...utils.image_utils import load_image_with_mask, prepare_mask, check_mask_for_errors
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

        if logger == None:
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

        if callback is not None:
            self.datacallback = callback

        if self.logging and hasattr(self.logger, "start_session"):
            self.logger.start_session()
            start_total_time = time.time()
            duration = (start_total_time - self.script_start_time) * 1000
            self.logger.log(f"Script startup / model loading took {duration:.2f} ms")
        else:
            self.logging = False

        if settings_file:
            self.gen = DeforumGenerationObject.from_settings_file(settings_file)
        else:
            self.gen = DeforumGenerationObject(**kwargs)

        self.gen.update_from_kwargs(**kwargs)

        setup_start = time.time()
        self.pre_setup()
        setup_end = time.time()
        duration = (setup_end - setup_start) * 1000
        if self.logging:
            self.logger.log(f"pre_setup took {duration:.2f} ms")

            setup_start = time.time()
        self.setup()
        if self.logging:
            setup_end = time.time()
            duration = (setup_end - setup_start) * 1000
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


        self.pbar = tqdm(total=self.gen.max_frames, desc="Processing", position=0, leave=True)
        # PREP LOOP
        for fn in self.prep_fns:
            start_time = time.time()
            fn(self)
            if self.logging:
                end_time = time.time()
                duration = (end_time - start_time) * 1000
                self.logger.log(f"{fn.__name__} took {duration:.2f} ms")

        while self.gen.frame_idx + 1 <= self.gen.max_frames:
            # MAIN LOOP
            frame_start = time.time()
            for fn in self.shoot_fns:
                start_time = time.time()
                with torch.inference_mode():
                    with torch.no_grad():
                        fn(self)
                if self.logging:
                    end_time = time.time()
                    duration = (end_time - start_time) * 1000
                    self.logger.log(f"{fn.__name__} took {duration:.2f} ms")
            self.pbar.update(self.gen.turbo_steps)
            if self.logging:
                duration = (time.time() - frame_start) * 1000
                self.logger.log(f"----------------------------- Frame {self.gen.frame_idx + 1} took {duration:.2f} ms")
        self.pbar.close()

        # POST LOOP
        for fn in self.post_fns:
            start_time = time.time()
            fn(self)
            if self.logging:
                duration = (time.time() - start_time) * 1000
                self.logger.log(f"{fn.__name__} took {duration:.2f} ms")
        if self.logging:
            total_duration = (time.time() - start_total_time) * 1000
            average_time_per_frame = total_duration / self.gen.max_frames
            self.logger.log(f"Total time taken: {total_duration:.2f} ms")
            self.logger.log(f"Average time per frame: {average_time_per_frame:.2f} ms")
            self.logger.close_session()
        return self.gen
    def pre_setup(self):
        frame_warp_modes = ['2D', '3D']
        hybrid_motion_modes = ['Affine', 'Perspective', 'Optical Flow']

        self.gen.max_frames += 1

        if self.gen.animation_mode in frame_warp_modes:
            # handle hybrid video generation
            if self.gen.hybrid_composite != 'None' or self.gen.hybrid_motion in hybrid_motion_modes:
                _, _, self.gen.inputfiles = hybrid_generation(self.gen, self.gen, self.gen)
                self.gen.hybrid_frame_path = os.path.join(self.gen.outdir, 'hybridframes')

        if int(self.gen.seed) == -1:
            self.gen.seed = secrets.randbelow(18446744073709551615)
        self.gen.keys = DeformAnimKeys(self.gen, self.gen.seed)
        self.gen.loopSchedulesAndData = LooperAnimKeys(self.gen, self.gen, self.gen.seed)
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
        self.gen.predict_depths = (
                                     self.gen.animation_mode == '3D' and self.gen.use_depth_warping) or self.gen.save_depth_maps
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
                                     depth_algorithm=self.gen.depth_algorithm, Width=self.gen.W,
                                     Height=self.gen.H,
                                     midas_weight=self.gen.midas_weight)
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

    def setup(self, *args, **kwargs) -> None:
        """
        Set up the list of functions to be executed during the main loop of the animation pipeline.

        This method populates the `shoot_fns` list with functions based on the configuration set in the `gen` object.
        Certain functions are added to the list based on the conditions provided by the attributes of the `gen` object.
        Additionally, post-processing functions can be added to the `post_fns` list.
        """
        self.reset()

        hybrid_available = self.gen.hybrid_composite != 'None' or self.gen.hybrid_motion in ['Optical Flow', 'Affine', 'Perspective']

        turbo_steps = self.gen.get('turbo_steps', 1)
        if turbo_steps > 1:
            self.shoot_fns.append(make_cadence_frames)
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

        self.shoot_fns.append(get_generation_params)

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
        if self.gen.max_frames > 1:
            self.post_fns.append(save_video_cls)


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

    def generate(self):
        """
        Generates an image or animation using the given prompts, settings, and generator.

        This method sets up the necessary arguments, handles conditional configurations, and then
        uses the provided generator to produce the output.

        Returns:
            processed (Image): The generated image or animation frame.
        """
        assert self.gen.prompt is not None

        # Setup the pipeline
        # p = get_webui_sd_pipeline(args, root, frame)
        prompt, negative_prompt = split_weighted_subprompts(self.gen.prompt, self.gen.frame_idx, self.gen.max_frames)

        # print("DEFORUM CONDITIONING INTERPOLATION")

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

        next_prompt, blend_value = get_next_prompt_and_blend(self.gen.frame_idx, self.gen.prompt_series)
        # print("DEBUG", next_prompt, blend_value)

        # blend_value = 1.0
        # next_prompt = ""
        if not self.gen.use_init and self.gen.strength > 0 and self.gen.strength_0_no_init:
            self.gen.strength = 0
        processed = None
        mask_image = None
        init_image = None
        image_init0 = None

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
                                      shape=(self.gen.W, self.gen.H),
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
        if self.gen.scheduled_sampler_name is not None:
            if self.gen.scheduled_sampler_name in auto_to_comfy.keys():
                self.gen.sampler_name = auto_to_comfy[self.gen.sampler_name]["sampler"]
                self.gen.scheduler = auto_to_comfy[self.gen.sampler_name]["scheduler"]
            # else:
            #     raise RuntimeError(
            #         f"Sampler name '{sampler_name}' is invalid. Please check the available sampler list in the 'Run' tab")

        # if self.gen.checkpoint is not None:
        #    info = sd_models.get_closet_checkpoint_match(self.gen.checkpoint)
        #    if info is None:
        #        raise RuntimeError(f"Unknown checkpoint: {self.gen.checkpoint}")
        #    sd_models.reload_model_weights(info=info)

        if self.gen.init_sample is not None:
            # TODO: cleanup init_sample remains later
            img = self.gen.init_sample
            init_image = img
            image_init0 = img
            if self.gen.use_looper and isJson(self.gen.imagesToKeyframe) and self.gen.animation_mode in ['2D', '3D']:
                init_image = Image.blend(init_image, init_image2, blendFactor)
                correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
                color_corrections = [correction_colors]

        # this is the first pass
        elif (self.gen.use_looper and self.gen.animation_mode in ['2D', '3D']) or (
                self.gen.use_init and ((self.gen.init_image != None and self.gen.init_image != ''))):
            init_image, mask_image = load_image_with_mask(image_init0,  # initial init image
                                              shape=(self.gen.W, self.gen.H),
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

            self.genstrength = 1.0 if init_image is None else self.gen.strength

            cnet_image = None
            input_file = os.path.join(self.gen.outdir, 'inputframes',
                                      get_frame_name(self.gen.video_init_path) + f"{self.gen.frame_idx:09}.jpg")

            # if os.path.isfile(input_file):
            #     input_frame = Image.open(input_file)
            #     cnet_image = get_canny_image(input_frame)
            #     cnet_image = ImageOps.invert(cnet_image)

            if prompt == "!reset!":
                self.gen.init_image = None
                self.genstrength = 1.0
                prompt = next_prompt

            if negative_prompt == "":
                negative_prompt = self.gen.animation_prompts_negative

            gen_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": self.gen.steps,
                "seed": self.gen.seed,
                "scale": self.gen.scale,
                "strength": self.genstrength,
                "init_image": init_image,
                "width": self.gen.W,
                "height": self.gen.H,
                "cnet_image": cnet_image,
                "next_prompt": next_prompt,
                "prompt_blend": blend_value,
                "scheduler":self.gen.scheduler,
                "sampler_name":self.gen.sampler_name
            }
            if self.gen.frame_idx == 0:
                gen_args["reset_noise"] = True

            # print(f"DEFORUM GEN ARGS: [{gen_args}] ")

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
                                    (self.gen.W, self.gen.H),
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
                "width": self.gen.W,
                "height": self.gen.H,
                "cnet_image": cnet_image,
                "next_prompt": next_prompt,
                "prompt_blend": blend_value
            }

            #print(f"DEFORUM GEN ARGS: [{gen_args}] ")

            if self.gen.enable_subseed_scheduling:
                gen_args["subseed"] = self.gen.subseed
                gen_args["subseed_strength"] = self.gen.subseed_strength
                gen_args["seed_resize_from_h"] = self.gen.seed_resize_from_h
                gen_args["seed_resize_from_w"] = self.gen.seed_resize_from_w


            processed = self.generator(**gen_args)

        if self.gen.first_frame == None:
            self.gen.first_frame = processed

        return processed
