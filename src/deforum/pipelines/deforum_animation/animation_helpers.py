import copy
import gc
import math
import os
import random
#from multiprocessing import Process
from typing import Any, Union, Tuple, Optional

import PIL.Image
import cv2
import numexpr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageChops

from deforum.utils.blocking_file_list import BlockingFileList

from ... import FILMInterpolator
from ...generators.deforum_flow_generator import (get_flow_for_hybrid_motion_prev,
                                                  get_flow_for_hybrid_motion,
                                                  abs_flow_to_rel_flow,
                                                  rel_flow_to_abs_flow, get_flow_from_images)
from ...generators.deforum_noise_generator import add_noise
from ...models.RIFE.rife_model import RIFEInterpolator
from ...pipeline_utils import next_seed
from ...utils import py3d_tools as p3d
from ...utils.constants import root_path, config
from ...utils.deforum_framewarp_utils import (get_flip_perspective_matrix,
                                              flip_3d_perspective,
                                              transform_image_3d_new,
                                              anim_frame_warp)
from ...utils.deforum_hybrid_animation import get_matrix_for_hybrid_motion, get_matrix_for_hybrid_motion_prev, \
    hybrid_composite
from ...utils.image_utils import (load_image,
                                  autocontrast_grayscale,
                                  image_transform_ransac,
                                  image_transform_optical_flow,
                                  unsharp_mask,
                                  get_mask_from_file, do_overlay_mask, save_image, maintain_colors,
                                  compose_mask_with_check)
from ...utils.string_utils import check_is_number, prepare_prompt
from ...utils.subtitle_handler import init_srt_file, format_animation_params, write_frame_subtitle
from ...utils.video_frame_utils import (get_frame_name,
                                        get_next_frame)
from ...utils.video_save_util import save_as_h264

from deforum.utils.logging_config import logger


def anim_frame_warp_cls(cls: Any) -> None:
    """
    Adjusts the animation frame warp for the given class instance based on various conditions.

    This function is an element of an animation pipeline that handles 2D/3D morphs on the given frame.
    It modifies parameters within the passed class instance based on the generation parameters object (cls.gen).

    Args:
        cls (Any): The class instance that contains generation parameters and needs frame warp adjustments.
                   This instance should have attributes like gen.prev_img, gen.use_depth_warping, etc.

    Returns:
        None
    """
    if cls.gen.prev_img is not None:
        cls.gen.mask = None
        if cls.gen.use_depth_warping:
            if cls.gen.depth is None and cls.depth_model is not None:
                if cls.depth_model.device != 'cuda':
                    cls.depth_model.to('cuda')
                    cls.depth_model.device = 'cuda'
                with torch.no_grad():
                    cls.gen.depth = cls.depth_model.predict(cls.gen.prev_img, cls.gen.midas_weight, True)
        else:
            cls.gen.depth = None
        if cls.gen.animation_mode == '2D':
            cls.gen.prev_img = anim_frame_warp_2d_cls(cls, cls.gen.prev_img)
        else:  # '3D'
            cls.gen.prev_img, cls.gen.mask = anim_frame_warp_3d_cls(cls, cls.gen.prev_img)
    return


def anim_frame_warp_cls_image(cls: Any, image: Union[None, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Adjusts the animation frame warp for a given image and class instance.

    Args:
        cls: The class instance containing generation parameters.
        image: The image to be processed.

    Returns:
        Tuple containing the processed image and its mask.
    """

    cls.gen.mask = None

    if image is not None:

        if cls.gen.use_depth_warping:
            if cls.gen.depth is None and cls.depth_model is not None:
                cls.gen.depth = cls.depth_model.predict(image, cls.gen.midas_weight, cls.gen.half_precision)
        else:
            cls.gen.depth = None

        if cls.gen.animation_mode == '2D':
            cls.gen.image = anim_frame_warp_2d_cls(cls, image)
        else:  # '3D'
            cls.gen.image, cls.gen.mask = anim_frame_warp_3d_cls(cls, image)
    return cls.gen.image, cls.gen.mask


def anim_frame_warp_2d_cls(cls: Any, image: Union[None, Any]) -> Any:
    """
    Adjusts the 2D animation frame warp for a given image and class instance based on transformation parameters.

    Args:
        cls: The class instance containing generation parameters and transformation keys.
        image: The image to be processed.

    Returns:
        Processed image after 2D transformation.
    """
    angle = cls.gen.keys.angle_series[cls.gen.frame_idx]
    zoom = cls.gen.keys.zoom_series[cls.gen.frame_idx]
    translation_x = cls.gen.keys.translation_x_series[cls.gen.frame_idx]
    translation_y = cls.gen.keys.translation_y_series[cls.gen.frame_idx]
    transform_center_x = cls.gen.keys.transform_center_x_series[cls.gen.frame_idx]
    transform_center_y = cls.gen.keys.transform_center_y_series[cls.gen.frame_idx]
    center_point = (cls.gen.width * transform_center_x, cls.gen.height * transform_center_y)
    rot_mat = cv2.getRotationMatrix2D(center_point, angle, zoom)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    if cls.gen.enable_perspective_flip:
        bM = get_flip_perspective_matrix(cls.gen.width, cls.gen.height, cls.gen.keys, cls.gen.frame_idx)
        rot_mat = np.matmul(bM, rot_mat, trans_mat)
    else:
        rot_mat = np.matmul(rot_mat, trans_mat)
    return cv2.warpPerspective(
        image,
        rot_mat,
        (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_WRAP if cls.gen.border == 'wrap' else cv2.BORDER_REPLICATE
    )


def anim_frame_warp_3d_cls(cls: Any, image: Union[None, Any]) -> Tuple[Any, Any]:
    """
    Adjusts the 3D animation frame warp for a given image and class instance based on transformation parameters.

    Args:
        cls: The class instance containing generation parameters and transformation keys.
        image: The image to be processed.

    Returns:
        Tuple containing the processed image after 3D transformation and its mask.
    """
    TRANSLATION_SCALE = 1.0 / 200.0  # matches Disco
    translate_xyz = [
        -cls.gen.keys.translation_x_series[cls.gen.frame_idx] * TRANSLATION_SCALE,
        cls.gen.keys.translation_y_series[cls.gen.frame_idx] * TRANSLATION_SCALE,
        -cls.gen.keys.translation_z_series[cls.gen.frame_idx] * TRANSLATION_SCALE
    ]
    rotate_xyz = [
        math.radians(cls.gen.keys.rotation_3d_x_series[cls.gen.frame_idx]),
        math.radians(cls.gen.keys.rotation_3d_y_series[cls.gen.frame_idx]),
        math.radians(cls.gen.keys.rotation_3d_z_series[cls.gen.frame_idx])
    ]
    if cls.gen.enable_perspective_flip:
        image = flip_3d_perspective(cls.gen, image, cls.gen.keys, cls.gen.frame_idx)
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device="cuda"), "XYZ").unsqueeze(0)


    result = transform_image_3d_new(torch.device('cuda'), image, cls.gen.depth, rot_mat, translate_xyz,
                                          cls.gen, cls.gen.keys, cls.gen.frame_idx)
    return result, None


def anim_frame_warp_3d_direct(cls, image, x, y, z, rx, ry, rz):
    TRANSLATION_SCALE = 1.0 / 200.0  # matches Disco
    translate_xyz = [
        -x * TRANSLATION_SCALE,
        y * TRANSLATION_SCALE,
        -z * TRANSLATION_SCALE
    ]
    rotate_xyz = [
        math.radians(rx),
        math.radians(ry),
        math.radians(rz)
    ]
    if cls.gen.enable_perspective_flip:
        image = flip_3d_perspective(cls.gen, image, cls.gen.keys, cls.gen.frame_idx)
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device="cuda"), "XYZ").unsqueeze(0)
    result, mask = transform_image_3d_new(torch.device('cuda'), image, cls.gen.depth, rot_mat, translate_xyz,
                                          cls.gen, cls.gen.keys, cls.gen.frame_idx)
    return result, mask

def hybrid_composite_cls(cls: Any) -> None:
    """
    Creates a hybrid composite frame for the given class instance based on various conditions and transformation
    parameters.

    Args:
        cls: The class instance containing generation parameters, image paths, transformation keys, and other settings.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if cls.gen.prev_img is not None:

        video_frame_path = os.path.join(cls.gen.outdir, 'inputframes')

        if config.allow_blocking_input_frame_lists:
            inputfiles = BlockingFileList(video_frame_path, cls.gen.max_frames)
            video_frame = inputfiles[cls.gen.frame_idx]
        else: 
            video_frame = os.path.join(cls.gen.outdir, 'inputframes',
                                    get_frame_name(cls.gen.video_init_path) + f"{cls.gen.frame_idx:09}.jpg")
        video_depth_frame = os.path.join(cls.gen.outdir, 'hybridframes',
                                         get_frame_name(
                                             cls.gen.video_init_path) + f"_vid_depth{cls.gen.frame_idx:09}.jpg")
        depth_frame = os.path.join(cls.gen.outdir, f"{cls.gen.timestring}_depth_{cls.gen.frame_idx - 1:09}.png")
        mask_frame = os.path.join(cls.gen.outdir, 'hybridframes',
                                  get_frame_name(cls.gen.video_init_path) + f"_mask{cls.gen.frame_idx:09}.jpg")
        comp_frame = os.path.join(cls.gen.outdir, 'hybridframes',
                                  get_frame_name(cls.gen.video_init_path) + f"_comp{cls.gen.frame_idx:09}.jpg")
        prev_frame = os.path.join(cls.gen.outdir, 'hybridframes',
                                  get_frame_name(cls.gen.video_init_path) + f"_prev{cls.gen.frame_idx:09}.jpg")
        cls.gen.prev_img = cv2.cvtColor(cls.gen.prev_img, cv2.COLOR_BGR2RGB)
        prev_img_hybrid = Image.fromarray(cls.gen.prev_img)
        if cls.gen.hybrid_use_init_image:
            video_image = load_image(cls.gen.init_image, cls.gen.init_image_box)
        else:
            video_image = Image.open(video_frame)
        video_image = video_image.resize((cls.gen.width, cls.gen.height), Image.LANCZOS)
        hybrid_mask = None

        # composite mask types
        if cls.gen.hybrid_comp_mask_type == 'Depth':  # get depth from last generation
            hybrid_mask = Image.open(depth_frame)
        elif cls.gen.hybrid_comp_mask_type == 'Video Depth':  # get video depth
            video_depth = cls.depth_model.predict(np.array(video_image), cls.gen.midas_weight, cls.gen.half_precision)
            cls.depth_model.save(video_depth_frame, video_depth)
            hybrid_mask = Image.open(video_depth_frame)
        elif cls.gen.hybrid_comp_mask_type == 'Blend':  # create blend mask image
            hybrid_mask = Image.blend(ImageOps.grayscale(prev_img_hybrid), ImageOps.grayscale(video_image),
                                      cls.gen.hybrid_comp_schedules['mask_blend_alpha'])
        elif cls.gen.hybrid_comp_mask_type == 'Difference':  # create difference mask image
            hybrid_mask = ImageChops.difference(ImageOps.grayscale(prev_img_hybrid), ImageOps.grayscale(video_image))

        # optionally invert mask, if mask type is defined
        if cls.gen.hybrid_comp_mask_inverse and cls.gen.hybrid_comp_mask_type != "None":
            hybrid_mask = ImageOps.invert(hybrid_mask)

        # if a mask type is selected, make composition
        if hybrid_mask is None:
            hybrid_comp = video_image
        else:
            # ensure grayscale
            hybrid_mask = ImageOps.grayscale(hybrid_mask)
            # equalization before
            if cls.gen.hybrid_comp_mask_equalize in ['Before', 'Both']:
                hybrid_mask = ImageOps.equalize(hybrid_mask)
                # contrast
            hybrid_mask = ImageEnhance.Contrast(hybrid_mask).enhance(cls.gen.hybrid_comp_schedules['mask_contrast'])
            # auto contrast with cutoffs lo/hi
            if cls.gen.hybrid_comp_mask_auto_contrast:
                hybrid_mask = autocontrast_grayscale(np.array(hybrid_mask),
                                                     cls.gen.hybrid_comp_schedules['mask_auto_contrast_cutoff_low'],
                                                     cls.gen.hybrid_comp_schedules['mask_auto_contrast_cutoff_high'])
                hybrid_mask = Image.fromarray(hybrid_mask)
                hybrid_mask = ImageOps.grayscale(hybrid_mask)
            if cls.gen.hybrid_comp_save_extra_frames:
                hybrid_mask.save(mask_frame)
                # equalization after
            if cls.gen.hybrid_comp_mask_equalize in ['After', 'Both']:
                hybrid_mask = ImageOps.equalize(hybrid_mask)
                # do compositing and save
            hybrid_comp = Image.composite(prev_img_hybrid, video_image, hybrid_mask)
            if cls.gen.hybrid_comp_save_extra_frames:
                hybrid_comp.save(comp_frame)

        # final blend of composite with prev_img, or just a blend if no composite is selected
        hybrid_blend = Image.blend(prev_img_hybrid, hybrid_comp, cls.gen.hybrid_comp_schedules['alpha'])
        if cls.gen.hybrid_comp_save_extra_frames:
            hybrid_blend.save(prev_frame)

        cls.gen.prev_img = cv2.cvtColor(np.array(hybrid_blend), cv2.COLOR_RGB2BGR)

    # restore to np array and return
    return


def affine_persp_motion(cls: Any) -> None:
    """
    Applies affine or perspective motion transformation to the previous image of the given class instance.

    Args:
        cls: The class instance containing generation parameters, motion settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if cls.gen.frame_idx < 1:
        logger.info("Skipping optical flow motion for first frame.")
        return
    if cls.gen.hybrid_motion_use_prev_img and cls.gen.prev_img is not None:
        matrix = get_matrix_for_hybrid_motion_prev(cls.gen.frame_idx - 1, (cls.gen.width, cls.gen.height), cls.gen.inputfiles,
                                                   cls.gen.prev_img,
                                                   cls.gen.hybrid_motion)
    else:
        matrix = get_matrix_for_hybrid_motion(cls.gen.frame_idx - 1, (cls.gen.width, cls.gen.height), cls.gen.inputfiles,
                                              cls.gen.hybrid_motion)
    cls.gen.prev_img = image_transform_ransac(cls.gen.prev_img, matrix, cls.gen.hybrid_motion)
    return


def optical_flow_motion(cls: Any) -> None:
    """
    Applies optical flow motion transformation to the previous image of the given class instance.

    Args: cls: The class instance containing generation parameters, motion settings, optical flow methods,
    and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if cls.gen.frame_idx < 1:
        logger.info("Skipping optical flow motion for first frame.")
        return

    if cls.gen.prev_img is not None and cls.gen.inputfiles is not None:
        if cls.gen.hybrid_motion_use_prev_img:
            cls.gen.flow = get_flow_for_hybrid_motion_prev(cls.gen.frame_idx - 1, (cls.gen.width, cls.gen.height),
                                                           cls.gen.inputfiles,
                                                           cls.gen.hybrid_frame_path, cls.gen.prev_flow,
                                                           cls.gen.prev_img,
                                                           cls.gen.hybrid_flow_method, cls.raft_model,
                                                           cls.gen.hybrid_flow_consistency,
                                                           cls.gen.hybrid_consistency_blur,
                                                           cls.gen.hybrid_comp_save_extra_frames)


        else:
            cls.gen.flow = get_flow_for_hybrid_motion(cls.gen.frame_idx - 1, (cls.gen.width, cls.gen.height), cls.gen.inputfiles,
                                                      cls.gen.hybrid_frame_path,
                                                      cls.gen.prev_flow, cls.gen.hybrid_flow_method, cls.raft_model,
                                                      cls.gen.hybrid_flow_consistency,
                                                      cls.gen.hybrid_consistency_blur,
                                                      cls.gen.hybrid_comp_save_extra_frames)
        cls.gen.prev_img = image_transform_optical_flow(cls.gen.prev_img, cls.gen.flow,
                                                        cls.gen.hybrid_comp_schedules['flow_factor'])
        cls.gen.prev_flow = cls.gen.flow

    return


def color_match_cls(cls: Any) -> None:
    """
    Matches the color of the previous image to a reference sample in the given class instance.

    Args:
        cls: The class instance containing generation parameters, color match settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if cls.gen.color_match_sample is None and cls.gen.image is not None:
        cls.gen.color_match_sample = cls.gen.prev_img.copy()
    elif cls.gen.prev_img is not None:
        cls.gen.prev_img = maintain_colors(cls.gen.prev_img, cls.gen.color_match_sample, cls.gen.color_coherence)
    return


def set_contrast_image(cls: Any) -> None:
    """
    Adjusts the contrast of the previous image in the given class instance.

    Args:
        cls: The class instance containing generation parameters, contrast settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if cls.gen.prev_img is not None:
        # intercept and override to grayscale
        if cls.gen.color_force_grayscale:
            cls.gen.prev_img = cv2.cvtColor(cls.gen.prev_img, cv2.COLOR_BGR2GRAY)
            cls.gen.prev_img = cv2.cvtColor(cls.gen.prev_img, cv2.COLOR_GRAY2BGR)

        # apply scaling
        cls.gen.contrast_image = (cls.gen.prev_img * cls.gen.contrast).round().astype(np.uint8)
        # anti-blur
        if cls.gen.amount > 0:
            cls.gen.contrast_image = unsharp_mask(cls.gen.contrast_image, (cls.gen.kernel, cls.gen.kernel),
                                                  cls.gen.sigma, cls.gen.amount, cls.gen.threshold,
                                                  cls.gen.mask_image if cls.gen.use_mask else None)
    return


def handle_noise_mask(cls: Any) -> None:
    """
    Composes a noise mask for the contrast image in the given class instance.

    Args:
        cls: The class instance containing generation parameters, noise mask settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    cls.gen.noise_mask = compose_mask_with_check(cls.gen, cls.gen, cls.gen.noise_mask_seq, cls.gen.noise_mask_vals,
                                                 Image.fromarray(
                                                     cv2.cvtColor(cls.gen.contrast_image, cv2.COLOR_BGR2RGB)))
    return


def add_noise_cls(cls: Any) -> None:
    """
    Adds noise to the contrast image in the given class instance.

    Args:
        cls: The class instance containing generation parameters, noise settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if cls.gen.prev_img is not None:
        noised_image = add_noise(cls.gen.contrast_image, cls.gen.noise, cls.gen.seed, cls.gen.noise_type,
                                 (cls.gen.perlin_w, cls.gen.perlin_h, cls.gen.perlin_octaves,
                                  cls.gen.perlin_persistence),
                                 cls.gen.noise_mask, cls.gen.invert_mask)

        # use transformed previous frame as init for current
        cls.gen.use_init = True
        cls.gen.init_sample = Image.fromarray(cv2.cvtColor(noised_image, cv2.COLOR_BGR2RGB))
        cls.gen.prev_img = noised_image
        cls.gen.strength = max(0.0, min(1.0, cls.gen.strength))
    return


def get_generation_params(cls: Any) -> None:
    """
    Fetches and sets generation parameters for the given class instance based on various conditions and schedules.

    Args:
        cls: The class instance containing various generation parameters, schedules, settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """

    frame_idx = cls.gen.frame_idx
    keys = cls.gen.keys

    # logger.info(f"\033[36mAnimation frame: \033[0m{frame_idx}/{cls.gen.max_frames}  ")

    cls.gen.noise = keys.noise_schedule_series[frame_idx]
    cls.gen.strength = keys.strength_schedule_series[frame_idx]
    cls.gen.scale = keys.cfg_scale_schedule_series[frame_idx]
    cls.gen.contrast = keys.contrast_schedule_series[frame_idx]
    cls.gen.kernel = int(keys.kernel_schedule_series[frame_idx])
    cls.gen.sigma = keys.sigma_schedule_series[frame_idx]
    cls.gen.amount = keys.amount_schedule_series[frame_idx]
    cls.gen.threshold = keys.threshold_schedule_series[frame_idx]
    cls.gen.cadence_flow_factor = keys.cadence_flow_factor_schedule_series[frame_idx]
    cls.gen.redo_flow_factor = keys.redo_flow_factor_schedule_series[frame_idx]
    cls.gen.hybrid_comp_schedules = {
        "alpha": keys.hybrid_comp_alpha_schedule_series[frame_idx],
        "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[frame_idx],
        "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[frame_idx],
        "mask_auto_contrast_cutoff_low": int(
            keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[frame_idx]),
        "mask_auto_contrast_cutoff_high": int(
            keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[frame_idx]),
        "flow_factor": keys.hybrid_flow_factor_schedule_series[frame_idx]
    }
    cls.gen.scheduled_sampler_name = None
    cls.gen.scheduled_clipskip = None
    cls.gen.scheduled_noise_multiplier = None
    cls.gen.scheduled_ddim_eta = None
    cls.gen.scheduled_ancestral_eta = None

    cls.gen.mask_seq = None
    cls.gen.noise_mask_seq = None
    if cls.gen.enable_steps_scheduling and keys.steps_schedule_series[frame_idx] is not None:
        cls.gen.steps = int(keys.steps_schedule_series[frame_idx])
    if cls.gen.enable_sampler_scheduling and keys.sampler_schedule_series[frame_idx] is not None:
        cls.gen.scheduled_sampler_name = keys.sampler_schedule_series[frame_idx].casefold()
    if cls.gen.enable_clipskip_scheduling and keys.clipskip_schedule_series[frame_idx] is not None:
        cls.gen.scheduled_clipskip = int(keys.clipskip_schedule_series[frame_idx])
    if cls.gen.enable_noise_multiplier_scheduling and keys.noise_multiplier_schedule_series[
        frame_idx] is not None:
        cls.gen.scheduled_noise_multiplier = float(keys.noise_multiplier_schedule_series[frame_idx])
    if cls.gen.enable_ddim_eta_scheduling and keys.ddim_eta_schedule_series[frame_idx] is not None:
        cls.gen.scheduled_ddim_eta = float(keys.ddim_eta_schedule_series[frame_idx])
    if cls.gen.enable_ancestral_eta_scheduling and keys.ancestral_eta_schedule_series[frame_idx] is not None:
        cls.gen.scheduled_ancestral_eta = float(keys.ancestral_eta_schedule_series[frame_idx])
    if cls.gen.use_mask and keys.mask_schedule_series[frame_idx] is not None:
        cls.gen.mask_seq = keys.mask_schedule_series[frame_idx]
    if cls.gen.use_noise_mask and keys.noise_mask_schedule_series[frame_idx] is not None:
        cls.gen.noise_mask_seq = keys.noise_mask_schedule_series[frame_idx]

    if cls.gen.use_mask and not cls.gen.use_noise_mask:
        cls.gen.noise_mask_seq = cls.gen.mask_seq

    cls.gen.depth = None

    # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
    cls.gen.pix2pix_img_cfg_scale = float(cls.gen.keys.pix2pix_img_cfg_scale_series[cls.gen.frame_idx])

    # grab prompt for current frame
    cls.gen.prompt = cls.gen.prompt_series[cls.gen.frame_idx]

    if cls.gen.seed_behavior == 'schedule' or cls.parseq_adapter.manages_seed():
        cls.gen.seed = int(keys.seed_schedule_series[frame_idx])

    if cls.gen.enable_checkpoint_scheduling:
        cls.gen.checkpoint = cls.gen.keys.checkpoint_schedule_series[cls.gen.frame_idx]
    else:
        cls.gen.checkpoint = None

    # SubSeed scheduling
    if cls.gen.enable_subseed_scheduling:
        cls.gen.subseed = int(cls.gen.keys.subseed_schedule_series[cls.gen.frame_idx])
        cls.gen.subseed_strength = float(cls.gen.keys.subseed_strength_schedule_series[cls.gen.frame_idx])

    if cls.parseq_adapter.manages_seed():
        cls.gen.enable_subseed_scheduling = True
        cls.gen.subseed = int(keys.subseed_schedule_series[frame_idx])
        cls.gen.subseed_strength = keys.subseed_strength_schedule_series[frame_idx]

    if cls.parseq_adapter.manages_prompts():
        cls.gen.prompt = cls.parseq_adapter.anim_keys.prompts[frame_idx]

    # set value back into the prompt - prepare and report prompt and seed
    cls.gen.prompt = prepare_prompt(cls.gen.prompt, cls.gen.max_frames, cls.gen.seed, cls.gen.frame_idx)

    # grab init image for current frame
    if cls.gen.using_vid_init:
        init_frame = get_next_frame(cls.gen.outdir, cls.gen.video_init_path, cls.gen.frame_idx, False)
        # print(f"Using video init frame {init_frame}")
        cls.gen.init_image = init_frame
        cls.gen.init_image_box = None  # init_image_box not used in this case
        cls.gen.strength = max(0.0, min(1.0, cls.gen.strength))
    if cls.gen.use_mask_video:
        cls.gen.mask_file = get_mask_from_file(
            get_next_frame(cls.gen.outdir, cls.gen.video_mask_path, cls.gen.frame_idx, True),
            cls.gen)
        cls.gen.noise_mask = get_mask_from_file(
            get_next_frame(cls.gen.outdir, cls.gen.video_mask_path, cls.gen.frame_idx, True), cls.gen)

        cls.gen.mask_vals['video_mask'] = get_mask_from_file(
            get_next_frame(cls.gen.outdir, cls.gen.video_mask_path, cls.gen.frame_idx, True), cls.gen)

    if cls.gen.use_mask:
        cls.gen.mask_image = compose_mask_with_check(cls.gen, cls.gen, cls.gen.mask_seq, cls.gen.mask_vals,
                                                     cls.gen.init_sample) if cls.gen.init_sample is not None else None  # we need it only after the first frame anyway

    # setting up some arguments for the looper
    cls.gen.imageStrength = cls.gen.loopSchedulesAndData.image_strength_schedule_series[cls.gen.frame_idx]
    cls.gen.blendFactorMax = cls.gen.loopSchedulesAndData.blendFactorMax_series[cls.gen.frame_idx]
    cls.gen.blendFactorSlope = cls.gen.loopSchedulesAndData.blendFactorSlope_series[cls.gen.frame_idx]
    cls.gen.tweeningFrameSchedule = cls.gen.loopSchedulesAndData.tweening_frames_schedule_series[cls.gen.frame_idx]
    cls.gen.colorCorrectionFactor = cls.gen.loopSchedulesAndData.color_correction_factor_series[cls.gen.frame_idx]
    cls.gen.use_looper = cls.gen.loopSchedulesAndData.use_looper
    cls.gen.imagesToKeyframe = cls.gen.loopSchedulesAndData.imagesToKeyframe

    # if 'img2img_fix_steps' in opts.data and opts.data[
    #     "img2img_fix_steps"]:  # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
    #     opts.data["img2img_fix_steps"] = False
    # if scheduled_clipskip is not None:
    #     opts.data["CLIP_stop_at_last_layers"] = scheduled_clipskip
    # if scheduled_noise_multiplier is not None:
    #     opts.data["initial_noise_multiplier"] = scheduled_noise_multiplier
    # if scheduled_ddim_eta is not None:
    #     opts.data["eta_ddim"] = scheduled_ddim_eta
    # if scheduled_ancestral_eta is not None:
    #     opts.data["eta_ancestral"] = scheduled_ancestral_eta

    # if cls.gen.animation_mode == '3D' and (cmd_opts.lowvram or cmd_opts.medvram):
    #     if predict_depths: depth_model.to('cpu')
    #     devices.torch_gc()
    #     lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
    #     sd_hijack.model_hijack.hijack(sd_model)
    return


def optical_flow_redo(cls: Any) -> None:
    """
    Applies optical flow redo transformation before generation based on given parameters and conditions.

    Args:
        cls: The class instance containing generation parameters, optical flow settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    optical_flow_redo_generation = cls.gen.optical_flow_redo_generation  # if not cls.gen.motion_preview_mode else 'None'

    # optical flow redo before generation
    if optical_flow_redo_generation != 'None' and cls.gen.prev_img is not None and cls.gen.strength > 0:
        stored_seed = cls.gen.seed
        cls.gen.seed = random.randint(0, 2 ** 32 - 1)
        # print(
        #     f"Optical flow redo is diffusing and warping using {optical_flow_redo_generation} and seed {cls.gen.seed} optical flow before generation.")

        disposable_image = cls.generate()

        disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
        disposable_flow = get_flow_from_images(cls.gen.prev_img, disposable_image, optical_flow_redo_generation,
                                               cls.raft_model)
        disposable_image = cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB)
        disposable_image = image_transform_optical_flow(disposable_image, disposable_flow, cls.gen.redo_flow_factor)
        cls.gen.seed = stored_seed
        cls.gen.init_sample = Image.fromarray(disposable_image)
        del (disposable_image, disposable_flow, stored_seed)
        gc.collect()

    return


def diffusion_redo(cls: Any) -> None:
    """
    Applies diffusion redo transformation before the final generation based on given parameters and conditions.

    Args:
        cls: The class instance containing generation parameters, diffusion redo settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if int(cls.gen.diffusion_redo) > 0 and cls.gen.prev_img is not None and cls.gen.strength > 0 and not cls.gen.motion_preview_mode:
        stored_seed = cls.gen.seed
        for n in range(0, int(cls.gen.diffusion_redo)):
            # print(f"Redo generation {n + 1} of {int(cls.gen.diffusion_redo)} before final generation")
            cls.gen.seed = random.randint(0, 2 ** 32 - 1)
            disposable_image = cls.generate()
            disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
            # color match on last one only
            if n == int(cls.gen.diffusion_redo):
                disposable_image = maintain_colors(cls.gen.prev_img, cls.gen.color_match_sample,
                                                   cls.gen.color_coherence)
            cls.gen.seed = stored_seed
            cls.gen.init_sample = Image.fromarray(cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB))
        del (disposable_image, stored_seed)
        # gc.collect()

    return


def main_generate_with_cls(cls: Any) -> None:
    """
    Executes the main generation process for the given class instance.
    Args:
        cls: The class instance containing generation parameters and other attributes.
    Returns:
        None: Modifies the class instance attributes in place.
    """
    cls.gen.image = cls.generate()
    return


def post_hybrid_composite_cls(cls: Any) -> None:
    """
    Executes the post-generation hybrid compositing process for the given class instance.

    Args:
        cls: The class instance containing generation parameters, hybrid compositing settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    # do hybrid video after generation
    if cls.gen.frame_idx > 0 and cls.gen.hybrid_composite == 'After Generation':
        image = cv2.cvtColor(np.array(cls.gen.image), cv2.COLOR_RGB2BGR)
        cls.gen, image = hybrid_composite(cls.gen, cls.gen, cls.gen.frame_idx, image, cls.depth_model,
                                          cls.gen.hybrid_comp_schedules, cls.gen)
        cls.gen.image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return


def post_color_match_with_cls(cls: Any) -> None:
    """
    Executes the post-generation color matching process for the given class instance.

    Args:
        cls: The class instance containing generation parameters, color matching settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    # color matching on first frame is after generation, color match was collected earlier, so we do an extra generation to avoid the corruption introduced by the color match of first output
    if cls.gen.color_match_sample is not None and 'post' in cls.gen.color_match_at:
        if cls.gen.frame_idx == 0 and (cls.gen.color_coherence == 'Image' or (
                cls.gen.color_coherence == 'Video Input' and cls.gen.hybrid_available)):
            image = maintain_colors(cv2.cvtColor(np.array(cls.gen.image), cv2.COLOR_RGB2BGR), cls.gen.color_match_sample,
                                    cls.gen.color_coherence)
            cls.gen.image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif cls.gen.color_match_sample is not None and cls.gen.color_coherence != 'None' and not cls.gen.legacy_colormatch:
            image = maintain_colors(cv2.cvtColor(np.array(cls.gen.image), cv2.COLOR_RGB2BGR), cls.gen.color_match_sample,
                                    cls.gen.color_coherence)
            cls.gen.image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return


def overlay_mask_cls(cls: Any) -> None:
    """
    Overlays a mask onto the generated image in the given class instance.

    Args:
        cls: The class instance containing generation parameters, overlay mask settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    # intercept and override to grayscale
    if cls.gen.color_force_grayscale:
        image = ImageOps.grayscale(cls.gen.image)
        cls.gen.image = ImageOps.colorize(image, black="black", white="white")

    # overlay mask
    if cls.gen.overlay_mask and (cls.gen.use_mask_video or cls.gen.use_mask):
        cls.gen.image = do_overlay_mask(cls.gen, cls.gen, cls.gen.image, cls.gen.frame_idx)

    # on strength 0, set color match to generation
    if ((not cls.gen.legacy_colormatch and not cls.gen.use_init) or (
            cls.gen.legacy_colormatch and cls.gen.strength == 0)) and not cls.gen.color_coherence in ['Image',
                                                                                                      'Video Input']:
        cls.gen.color_match_sample = cv2.cvtColor(np.asarray(cls.gen.image), cv2.COLOR_RGB2BGR)
    return


def post_gen_cls(cls: Any) -> None:
    """
    Executes post-generation processes for the given class instance including saving, updating images,
    and handling depth maps.

    Args:
        cls: The class instance containing generation parameters, image data, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if cls.gen.frame_idx < cls.gen.max_frames:
        # cls.logger(f"                                   [ ENTERING POST GEN CLS ]", True)
        if cls.gen.turbo_steps == 1:
            cls.images.append(np.array(cls.gen.image))
        # cls.logger(f"                                   [ frame added ]", True)

        cls.gen.opencv_image = cv2.cvtColor(np.array(cls.gen.image), cv2.COLOR_RGB2BGR)
        # cls.gen.opencv_image = np.array(cls.gen.image)


        # cls.logger(f"                                   [ cvtColor completed ]", True)

        if not cls.gen.using_vid_init:
            cls.gen.prev_img = cls.gen.opencv_image
        #     # cls.logger(f"                                   [ updated prev_img ]", True)

        if cls.gen.turbo_steps > 1 and cls.gen.frame_idx > 0:
            cls.gen.turbo_prev_image, cls.gen.turbo_prev_frame_idx = cls.gen.turbo_next_image, cls.gen.turbo_next_frame_idx
            cls.gen.turbo_next_image, cls.gen.turbo_next_frame_idx = cls.gen.opencv_image, cls.gen.frame_idx
            cls.gen.frame_idx += cls.gen.turbo_steps
            # cls.logger(f"                                   [ turbo_steps > 1 block executed ]", True)
        else:
            filename = f"{cls.gen.timestring}_{cls.gen.frame_idx:09}.png"
            # cls.logger(f"                                   [ filename generated: {filename} ]", True)
            # cv2.imwrite("current_deforum_debug.png", cls.gen.opencv_image)
            if not cls.gen.store_frames_in_ram:
                # p = Process(target=save_image, args=(cls.gen.image, 'PIL', filename, cls.gen, cls.gen, cls.gen))
                # p.start()
                save_image(cls.gen.image, 'PIL', filename, cls.gen, cls.gen, cls.gen)
                cls.gen.image_paths.append(os.path.join(cls.gen.outdir, filename))

                # cls.logger(f"                                   [ image saved ]", True)

            if cls.gen.save_depth_maps:
                cls.gen.depth = cls.depth_model.predict(cls.gen.opencv_image, cls.gen.midas_weight,
                                                        cls.gen.half_precision)
                cls.depth_model.save(
                    os.path.join(cls.gen.outdir, f"{cls.gen.timestring}_depth_{cls.gen.frame_idx:09}.png"),
                    cls.gen.depth)
                # cls.logger(f"                                   [ depth map saved ]", True)

            cls.gen.frame_idx += 1
            # cls.logger(f"                                   [ frame_idx incremented ]", True)
        if cls.gen.turbo_steps < 2:
            done = cls.datacallback({"image": cls.gen.image, "operation_id":cls.gen.operation_id, "frame_idx":cls.gen.frame_idx})
        # cls.logger(f"                                   [ datacallback executed ]", True)

        cls.gen.seed = next_seed(cls.gen, cls.gen)
        # cls.logger(f"                                   [ next seed generated ]", True)

    # cls.logger(f"                                   [ EXITING POST GEN CLS ]", True)
    cls.gen.image_count = len(cls.images)
    return


def generate_interpolated_frames(cls):
    
    turbo_prev_image = copy.deepcopy(cls.gen.turbo_prev_image)
    turbo_next_image = copy.deepcopy(cls.gen.turbo_next_image)
    turbo_prev_frame_idx = copy.deepcopy(cls.gen.turbo_prev_frame_idx)
    turbo_next_frame_idx = copy.deepcopy(cls.gen.turbo_next_frame_idx)
    
    # emit in-between frames
    if cls.gen.turbo_steps > 1 and cls.gen.frame_idx > 1:
        tween_frame_start_idx = max(cls.gen.start_frame, cls.gen.frame_idx - cls.gen.turbo_steps)
        cadence_flow = None
        for tween_frame_idx in range(tween_frame_start_idx, cls.gen.frame_idx):
            # update progress during cadence
            # state.job = f"frame {tween_frame_idx + 1}/{cls.gen.max_frames}"
            # state.job_no = tween_frame_idx + 1
            # cadence vars
            tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(cls.gen.frame_idx - tween_frame_start_idx)
            advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
            advance_next = tween_frame_idx > turbo_next_frame_idx

            # optical flow cadence setup before animation warping
            # if cls.gen.animation_mode in ['2D', '3D'] and cls.gen.optical_flow_cadence != 'None':
            if cls.gen.optical_flow_cadence != 'None':
                if cls.gen.keys.strength_schedule_series[tween_frame_start_idx] > 0:
                    if cadence_flow is None and turbo_prev_image is not None and turbo_next_image is not None:
                        cadence_flow = get_flow_from_images(turbo_prev_image, turbo_next_image,
                                                            cls.gen.optical_flow_cadence, cls.raft_model) / 2
                        turbo_next_image = image_transform_optical_flow(turbo_next_image, -cadence_flow, 1)

            # if opts.data.get("deforum_save_gen_info_as_srt"):
            #     params_to_print = opts.data.get("deforum_save_gen_info_as_srt_params", ['Seed'])
            #     params_string = format_animation_params(cls.gen.keys, cls.gen.prompt_series, tween_frame_idx, params_to_print)
            #     write_frame_subtitle(srt_filename, tween_frame_idx, srt_frame_duration,
            #                          f"F#: {tween_frame_idx}; Cadence: {tween < 1.0}; Seed: {cls.gen.seed}; {params_string}")
            #     params_string = None

            logger.info(
                f"Creating in-between {'' if cadence_flow is None else cls.gen.optical_flow_cadence + ' optical flow '}cadence frame: {tween_frame_idx}; tween:{tween:0.2f};")

            if cls.depth_model is not None:
                assert (turbo_next_image is not None)
                with torch.inference_mode():
                    depth = cls.depth_model.predict(turbo_next_image, cls.gen.midas_weight, cls.gen.half_precision)
            if cls.gen.animation_mode in ["3D", "2D"]:
                if advance_prev:
                    turbo_prev_image, _, _ = anim_frame_warp(turbo_prev_image, cls.gen, cls.gen, cls.gen.keys, tween_frame_idx,
                                                          cls.depth_model, depth=depth, device=cls.gen.device,
                                                          half_precision=cls.gen.half_precision)
                if advance_next:
                    turbo_next_image, _, _ = anim_frame_warp(turbo_next_image, cls.gen, cls.gen, cls.gen.keys, tween_frame_idx,
                                                          cls.depth_model, depth=depth, device=cls.gen.device,
                                                          half_precision=cls.gen.half_precision)

            # hybrid video motion - warps turbo_prev_image or turbo_next_image to match motion
            if tween_frame_idx > 0:
                if cls.gen.hybrid_motion in ['Affine', 'Perspective']:
                    if cls.gen.hybrid_motion_use_prev_img:
                        matrix = get_matrix_for_hybrid_motion_prev(tween_frame_idx - 1, (cls.gen.width, cls.gen.height), cls.gen.inputfiles,
                                                                   prev_img, cls.gen.hybrid_motion)
                        if advance_prev:
                            turbo_prev_image = image_transform_ransac(turbo_prev_image, matrix, cls.gen.hybrid_motion)
                        if advance_next:
                            turbo_next_image = image_transform_ransac(turbo_next_image, matrix, cls.gen.hybrid_motion)
                    else:
                        matrix = get_matrix_for_hybrid_motion(tween_frame_idx - 1, (cls.gen.width, cls.gen.height), cls.gen.inputfiles,
                                                              cls.gen.hybrid_motion)
                        if advance_prev:
                            turbo_prev_image = image_transform_ransac(turbo_prev_image, matrix, cls.gen.hybrid_motion)
                        if advance_next:
                            turbo_next_image = image_transform_ransac(turbo_next_image, matrix, cls.gen.hybrid_motion)
                if cls.gen.hybrid_motion in ['Optical Flow']:
                    if cls.gen.hybrid_motion_use_prev_img:
                        flow = get_flow_for_hybrid_motion_prev(tween_frame_idx - 1, (cls.gen.width, cls.gen.height), cls.gen.inputfiles,
                                                               cls.gen.hybrid_frame_path, cls.gen.prev_flow, prev_img,
                                                               cls.gen.hybrid_flow_method, cls.raft_model,
                                                               cls.gen.hybrid_flow_consistency,
                                                               cls.gen.hybrid_consistency_blur,
                                                               cls.gen.hybrid_comp_save_extra_frames)
                        if advance_prev:
                            turbo_prev_image = image_transform_optical_flow(turbo_prev_image, flow,
                                                                            cls.gen.hybrid_comp_schedules['flow_factor'])
                        if advance_next:
                            turbo_next_image = image_transform_optical_flow(turbo_next_image, flow,
                                                                            cls.gen.hybrid_comp_schedules['flow_factor'])
                        cls.gen.prev_flow = flow
                    else:
                        flow = get_flow_for_hybrid_motion(tween_frame_idx - 1, (cls.gen.width, cls.gen.height), cls.gen.inputfiles,
                                                          cls.gen.hybrid_frame_path, cls.gen.prev_flow, cls.gen.hybrid_flow_method,
                                                          cls.raft_model,
                                                          cls.gen.hybrid_flow_consistency,
                                                          cls.gen.hybrid_consistency_blur,
                                                          cls.gen.hybrid_comp_save_extra_frames)
                        if advance_prev:
                            turbo_prev_image = image_transform_optical_flow(turbo_prev_image, flow,
                                                                            cls.gen.hybrid_comp_schedules['flow_factor'])
                        if advance_next:
                            turbo_next_image = image_transform_optical_flow(turbo_next_image, flow,
                                                                            cls.gen.hybrid_comp_schedules['flow_factor'])
                        cls.gen.prev_flow = flow

            # do optical flow cadence after animation warping
            if cadence_flow is not None:
                cadence_flow = abs_flow_to_rel_flow(cadence_flow, cls.gen.width, cls.gen.height)
                cadence_flow, _, _ = anim_frame_warp(cadence_flow, cls.gen, cls.gen, cls.gen.keys, tween_frame_idx, cls.depth_model,
                                                  depth=depth, device=cls.gen.device, half_precision=cls.gen.half_precision)
                cadence_flow_inc = rel_flow_to_abs_flow(cadence_flow, cls.gen.width, cls.gen.height) * tween
                if advance_prev:
                    turbo_prev_image = image_transform_optical_flow(turbo_prev_image, cadence_flow_inc,
                                                                    cls.gen.cadence_flow_factor)
                if advance_next:
                    turbo_next_image = image_transform_optical_flow(turbo_next_image, cadence_flow_inc,
                                                                    cls.gen.cadence_flow_factor)

            turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

            if turbo_prev_image is not None and tween < 1.0:
                img = turbo_prev_image * (1.0 - tween) + turbo_next_image * tween
            else:
                img = turbo_next_image

            # intercept and override to grayscale
            if cls.gen.color_force_grayscale:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # overlay mask
            if cls.gen.overlay_mask and (cls.gen.use_mask_video or cls.gen.use_mask):
                img = do_overlay_mask(cls.gen, cls.gen, img, tween_frame_idx, True)

            # get prev_img during cadence
            prev_img = copy.deepcopy(img)


            # current image update for cadence frames (left commented because it doesn't currently update the preview)
            # state.current_image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

            # saving cadence frames
            filename = f"{cls.gen.timestring}_{tween_frame_idx:09}.png"
            cv2.imwrite(os.path.join(cls.gen.outdir, filename), img)
            # cv2.imwrite("current_cadence.png", img)
            cb_img = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
            cls.datacallback({"image":cb_img, "operation_id":cls.gen.operation_id, "frame_idx":cls.gen.frame_idx})
            cls.images.append(copy.deepcopy(cb_img))


            if cls.gen.save_depth_maps:
                cls.depth_model.save(os.path.join(cls.gen.outdir, f"{cls.gen.timestring}_depth_{tween_frame_idx:09}.png"), depth)
            cls.gen.prev_img = copy.deepcopy(prev_img)
            cls.gen.turbo_prev_image = copy.deepcopy(turbo_prev_image)
            cls.gen.turbo_next_image = copy.deepcopy(turbo_next_image)



def make_cadence_frames(cls: Any) -> None:
    """
    Generates intermediate frames between keyframes to create smoother animations.

    This function uses optical flow or hybrid motion methods to determine motion between frames.
    It then generates in-between frames (cadence frames) for smoother animations. The function also
    handles grayscale conversion, mask overlay, and saving cadence frames.

    Args:
        cls: The class instance containing generation parameters, image data, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if cls.gen.turbo_next_image is not None:
        if cls.gen.turbo_steps > 1:

            if isinstance(cls.gen.image, PIL.Image.Image):
                cls.gen.image = cv2.cvtColor(np.array(cls.gen.image), cv2.COLOR_RGB2BGR)
            if isinstance(cls.gen.prev_img, PIL.Image.Image):
                cls.gen.prev_img = cv2.cvtColor(np.array(cls.gen.prev_img), cv2.COLOR_RGB2BGR)
            if isinstance(cls.gen.turbo_next_image, PIL.Image.Image):
                cls.gen.turbo_next_image = cv2.cvtColor(np.array(cls.gen.turbo_next_image), cv2.COLOR_RGB2BGR)
            if hasattr(cls.gen, "img"):
                if isinstance(cls.gen.img, PIL.Image.Image):
                    cls.gen.img = cv2.cvtColor(np.array(cls.gen.img), cv2.COLOR_RGB2BGR)


            tween_frame_start_idx = max(cls.gen.start_frame, cls.gen.frame_idx - cls.gen.turbo_steps)
            cadence_flow = None
            for tween_frame_idx in range(tween_frame_start_idx, cls.gen.frame_idx):



                # update progress during cadence
                # state.job = f"frame {tween_frame_idx + 1}/{cls.gen.max_frames}"
                # state.job_no = tween_frame_idx + 1
                # cadence vars
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(
                    cls.gen.frame_idx - tween_frame_start_idx)
                advance_prev = cls.gen.turbo_prev_image is not None and tween_frame_idx > cls.gen.turbo_prev_frame_idx
                advance_next = tween_frame_idx > cls.gen.turbo_next_frame_idx

                # optical flow cadence setup before animation warping
                if cls.gen.animation_mode in ['2D', '3D'] and cls.gen.optical_flow_cadence != 'None':
                    if cls.gen.keys.strength_schedule_series[tween_frame_start_idx] > 0:
                        if cadence_flow is None and cls.gen.turbo_prev_image is not None and cls.gen.turbo_next_image is not None:
                            cadence_flow = get_flow_from_images(cls.gen.turbo_prev_image, cls.gen.turbo_next_image,
                                                                cls.gen.optical_flow_cadence, cls.raft_model) / 2
                            cls.gen.turbo_next_image = image_transform_optical_flow(cls.gen.turbo_next_image, -cadence_flow, 1)

                    # if opts.data.get("deforum_save_gen_info_as_srt"):
                    #     params_to_print = opts.data.get("deforum_save_gen_info_as_srt_params", ['Seed'])
                    #     params_string = format_animation_params(keys, prompt_series, tween_frame_idx, params_to_print)
                    #     write_frame_subtitle(srt_filename, tween_frame_idx, srt_frame_duration,
                    #                          f"F#: {tween_frame_idx}; Cadence: {tween < 1.0}; Seed: {cls.gen.seed}; {params_string}")
                    params_string = None

                # print(
                #     f"Creating in-between {'' if cadence_flow is None else cls.gen.optical_flow_cadence + ' optical flow '}cadence frame: {tween_frame_idx}; tween:{tween:0.2f};")

                if cls.depth_model is not None:
                    assert (cls.gen.turbo_next_image is not None)
                    depth = cls.depth_model.predict(cls.gen.turbo_next_image, cls.gen.midas_weight, cls.gen.half_precision)

                if advance_prev:
                    cls.gen.turbo_prev_image, _ = anim_frame_warp_cls_image(cls, cls.gen.turbo_prev_image)
                if advance_next:
                    cls.gen.turbo_next_image, _ = anim_frame_warp_cls_image(cls, cls.gen.turbo_prev_image)

                # hybrid video motion - warps turbo_prev_image or turbo_next_image to match motion
                if tween_frame_idx > 0:
                    if cls.gen.hybrid_motion in ['Affine', 'Perspective']:
                        if cls.gen.hybrid_motion_use_prev_img:
                            matrix = get_matrix_for_hybrid_motion_prev(tween_frame_idx - 1, (cls.gen.width, cls.gen.height),
                                                                       cls.gen.inputfiles, cls.gen.prev_img,
                                                                       cls.gen.hybrid_motion)
                            if advance_prev:
                                cls.gen.turbo_prev_image = image_transform_ransac(cls.gen.turbo_prev_image, matrix,
                                                                                  cls.gen.hybrid_motion)
                            if advance_next:
                                cls.gen.turbo_next_image = image_transform_ransac(cls.gen.turbo_next_image, matrix,
                                                                                  cls.gen.hybrid_motion)
                        else:
                            matrix = get_matrix_for_hybrid_motion(tween_frame_idx - 1, (cls.gen.width, cls.gen.height),
                                                                  cls.gen.inputfiles,
                                                                  cls.gen.hybrid_motion)
                            if advance_prev:
                                cls.gen.turbo_prev_image = image_transform_ransac(cls.gen.turbo_prev_image, matrix,
                                                                                  cls.gen.hybrid_motion)
                            if advance_next:
                                cls.gen.turbo_next_image = image_transform_ransac(cls.gen.turbo_next_image, matrix,
                                                                                  cls.gen.hybrid_motion)
                    if cls.gen.hybrid_motion in ['Optical Flow']:
                        if cls.gen.hybrid_motion_use_prev_img:
                            cls.gen.flow = get_flow_for_hybrid_motion_prev(tween_frame_idx - 1, (cls.gen.width, cls.gen.height),
                                                                           cls.gen.inputfiles,
                                                                           cls.gen.hybrid_frame_path, cls.gen.prev_flow,
                                                                           cls.gen.prev_img,
                                                                           cls.gen.hybrid_flow_method, cls.raft_model,
                                                                           cls.gen.hybrid_flow_consistency,
                                                                           cls.gen.hybrid_consistency_blur,
                                                                           cls.gen.hybrid_comp_save_extra_frames)
                            if advance_prev:
                                cls.gen.turbo_prev_image = image_transform_optical_flow(cls.gen.turbo_prev_image,
                                                                                        cls.gen.flow,
                                                                                        cls.gen.hybrid_comp_schedules[
                                                                                            'flow_factor'])
                            if advance_next:
                                cls.gen.turbo_next_image = image_transform_optical_flow(cls.gen.turbo_next_image,
                                                                                        cls.gen.flow,
                                                                                        cls.gen.hybrid_comp_schedules[
                                                                                            'flow_factor'])
                            cls.gen.prev_flow = cls.gen.flow
                        else:
                            cls.gen.flow = get_flow_for_hybrid_motion(tween_frame_idx - 1, (cls.gen.width, cls.gen.height),
                                                                      cls.gen.inputfiles,
                                                                      cls.gen.hybrid_frame_path, cls.gen.prev_flow,
                                                                      cls.gen.hybrid_flow_method, cls.raft_model,
                                                                      cls.gen.hybrid_flow_consistency,
                                                                      cls.gen.hybrid_consistency_blur,
                                                                      cls.gen.hybrid_comp_save_extra_frames)
                            if advance_prev:
                                cls.gen.turbo_prev_image = image_transform_optical_flow(cls.gen.turbo_prev_image,
                                                                                        cls.gen.flow,
                                                                                        cls.gen.hybrid_comp_schedules[
                                                                                            'flow_factor'])
                            if advance_next:
                                cls.gen.turbo_next_image = image_transform_optical_flow(cls.gen.turbo_next_image,
                                                                                        cls.gen.flow,
                                                                                        cls.gen.hybrid_comp_schedules[
                                                                                            'flow_factor'])
                            cls.gen.prev_flow = cls.gen.flow

                # do optical flow cadence after animation warping
                if cadence_flow is not None:
                    cadence_flow = abs_flow_to_rel_flow(cadence_flow, cls.gen.width, cls.gen.height)
                    cadence_flow, _, _ = anim_frame_warp(cadence_flow, cls.gen, cls.gen, cls.gen.keys, tween_frame_idx,
                                                         cls.depth_model,
                                                         depth=depth, device=cls.gen.device,
                                                         half_precision=cls.gen.half_precision)
                    cadence_flow_inc = rel_flow_to_abs_flow(cadence_flow, cls.gen.width, cls.gen.height) * tween
                    if advance_prev:
                        cls.gen.turbo_prev_image = image_transform_optical_flow(cls.gen.turbo_prev_image, cadence_flow_inc,
                                                                                cls.gen.cadence_flow_factor)
                    if advance_next:
                        cls.gen.turbo_next_image = image_transform_optical_flow(cls.gen.turbo_next_image, cadence_flow_inc,
                                                                                cls.gen.cadence_flow_factor)

                cls.gen.turbo_prev_frame_idx = cls.gen.turbo_next_frame_idx = tween_frame_idx

                if cls.gen.turbo_prev_image is not None and tween < 1.0:
                    cls.gen.img = cls.gen.turbo_prev_image * (1.0 - tween) + cls.gen.turbo_next_image * tween
                else:
                    cls.gen.img = cls.gen.turbo_next_image

                # intercept and override to grayscale
                if cls.gen.color_force_grayscale:
                    cls.gen.img = cv2.cvtColor(cls.gen.img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    cls.gen.img = cv2.cvtColor(cls.gen.img, cv2.COLOR_GRAY2BGR)

                    # overlay mask
                if cls.gen.overlay_mask and (cls.gen.use_mask_video or cls.gen.use_mask):
                    cls.gen.img = do_overlay_mask(cls.gen, cls.gen, cls.gen.img, tween_frame_idx, True)

                # get prev_img during cadence
                cls.gen.prev_img = cls.gen.img

                # current image update for cadence frames (left commented because it doesn't currently update the preview)
                # state.current_image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

                # saving cadence frames
                if not cls.gen.store_frames_in_ram:
                    filename = f"{cls.gen.timestring}_{tween_frame_idx:09}.png"
                    cv2.imwrite(os.path.join(cls.gen.outdir, filename), cls.gen.img)
                    cls.gen.image_paths.append(os.path.join(cls.gen.outdir, filename))


                callback_img = cv2.cvtColor(cls.gen.img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                done = cls.datacallback({"image": Image.fromarray(callback_img), "operation_id":cls.gen.operation_id, "frame_idx":cls.gen.frame_idx})
                # done = cls.datacallback({"image": Image.fromarray(cv2.cvtColor(cls.gen.img, cv2.COLOR_BGR2RGB))})
                cls.images.append(callback_img)

                if cls.gen.save_depth_maps:
                    cls.depth_model.save(
                        os.path.join(cls.gen.outdir, f"{cls.gen.timestring}_depth_{tween_frame_idx:09}.png"),
                        depth)




def color_match_video_input(cls: Any) -> None:
    """
    Matches the color of the generated image to the corresponding frame from the input video.

    If the current frame index is divisible by the specified interval (color_coherence_video_every_N_frames),
    the function fetches the corresponding frame from the input video and uses it as the reference
    for color matching. The fetched frame is resized to match the dimensions of the generated image
    and then used to set the color_match_sample attribute for later use in color coherence operations.

    Args:
        cls: The class instance containing generation parameters, image data, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if int(cls.gen.frame_idx) % int(cls.gen.color_coherence_video_every_N_frames) == 0:
        prev_vid_img = Image.open(os.path.join(cls.outdir, 'inputframes', get_frame_name(
            cls.video_init_path) + f"{cls.gen.frame_idx:09}.jpg"))
        cls.gen.prev_vid_img = prev_vid_img.resize((cls.W, cls.H), Image.LANCZOS)
        color_match_sample = np.asarray(cls.gen.prev_vid_img)
        cls.gen.color_match_sample = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2BGR)


def film_interpolate_cls(cls: Any) -> None:
    """
    Performs frame interpolation on a sequence of images using FILM and saves the interpolated video.

    The function calculates the number of in-between frames to add based on the frame_interpolation_x_amount attribute.
    It then uses FILM to interpolate between the original frames and generate the in-between frames.
    The interpolated video is then saved as an H264-encoded MP4 file.

    Args:
        cls: The class instance containing generation parameters, image data, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place and saves the interpolated video.
    """
    # "frame_interpolation_engine": "FILM",
    # "frame_interpolation_x_amount": 2,
    # "frame_interpolation_slow_mo_enabled": false,
    # "frame_interpolation_slow_mo_amount": 2,
    # "frame_interpolation_keep_imgs": false,
    # "frame_interpolation_use_upscaled": false,

    interpolator = FILMInterpolator()

    film_in_between_frames_count = calculate_frames_to_add(len(cls.images), cls.gen.frame_interpolation_x_amount)

    interpolated = interpolator(cls.images, film_in_between_frames_count)
    cls.images = interpolated
    cls.gen.image_paths = []
    return


def rife_interpolate_cls(cls: Any) -> None:
    """
    Performs frame interpolation on a sequence of images stored in cls.images using a custom RIFEInterpolator class.
    The interpolated images will replace the original sequence in cls.images.

    Args:
        cls: An instance of a class containing:
             - images: a list of PIL Image objects to interpolate.
             - frame_interpolation_x_amount: the number of frames to interpolate between each pair of images.

    Returns:
        None: Modifies the cls.images attribute in place.
    """

    # Initialize the interpolator
    interpolator = RIFEInterpolator()

    # Prepare an empty list to hold all the interpolated images
    new_images = []
    new_images.append(Image.fromarray(cls.images[0]))
    # Iterate through pairs of consecutive images in cls.images
    for i in range(len(cls.images) - 1):
        img1 = cls.images[i]
        img2 = cls.images[i + 1]

        # Interpolate between each pair
        interpolated_frames = interpolator.interpolate(np.array(img1), np.array(img2),
                                                       interp_amount=cls.gen.frame_interpolation_x_amount + 1)
        # Append all interpolated frames including the first of the pair but excluding the last one
        # The last frame of each segment should not be added to avoid duplication except for the final segment
        for i in interpolated_frames[:-1]:

            new_images.append(i)

    # After the loop, add the last frame of the last interpolated segment to complete the sequence
    if interpolated_frames:
        new_images.append(interpolated_frames[-1])  # Add the final image of the last interpolation batch

    # Print the number of interpolated frames for debugging
    print(f"Interpolated frame count: {len(new_images)}")

    # Replace the original images with the new, interpolated ones
    cls.images = new_images


    # Optionally clear image paths if no longer needed
    if hasattr(cls, 'gen') and hasattr(cls.gen, 'image_paths'):
        cls.gen.image_paths = []
    return


def save_video_cls(cls):
    dir_path = os.path.join(root_path, 'output/video')
    os.makedirs(dir_path, exist_ok=True)
    if cls.gen.timestring not in cls.gen.batch_name:
        name = f'{cls.gen.batch_name}_{cls.gen.timestring}'
    else:
        name = f'{cls.gen.batch_name}'
    output_filename_base = os.path.join(dir_path, name)
    #cls.gen.images = cls.images

    audio_path = None
    if hasattr(cls.gen, 'video_init_path'):
        audio_path = cls.gen.video_init_path

    fps = getattr(cls.gen, "fps", 24)  # Using getattr to simplify fetching attributes with defaults
    try:
        save_as_h264(cls.images, output_filename_base + ".mp4", audio_path=audio_path, fps=fps)
    except Exception as e:
        logger.error(f"save as h264 failed: {str(e)}")
    cls.gen.video_path = output_filename_base + ".mp4"


def calculate_frames_to_add(total_frames: int, interp_x: float) -> int:
    """
    Calculates the number of frames to add for interpolation based on the desired multiplier.

    Args:
        total_frames (int): The total number of original frames in the sequence.
        interp_x (float): The desired multiplier for frame interpolation.

    Returns:
        int: The number of frames to add between each original frame.
    """
    frames_to_add = (total_frames * interp_x - total_frames) / (total_frames - 1)
    return int(round(frames_to_add))

def cls_subtitle_handler(cls):
    if hasattr(cls.gen, "deforum_save_gen_info_as_srt"):
        if cls.gen.deforum_save_gen_info_as_srt:
            if cls.gen.frame_idx == 0 or not hasattr(cls.gen, 'srt_filename'):
                cls.gen.srt_filename = os.path.join(cls.gen.outdir, f"{cls.gen.timestring}.srt")
                cls.gen.srt_frame_duration = init_srt_file(cls.gen.srt_filename, cls.gen.fps)
            params_to_print = ["Trans X", "Trans Y", "Trans Z", "Str Sch", "Subseed Str Sch"]
            params_string = format_animation_params(cls.gen.keys, cls.gen.prompt_series, cls.gen.frame_idx, params_to_print)
            write_frame_subtitle(cls.gen.srt_filename, cls.gen.frame_idx, cls.gen.srt_frame_duration,
                                 f"F#: {cls.gen.frame_idx}; Cadence: false; Seed: {cls.gen.seed}; {params_string}")

class DeforumAnimKeys():
    def __init__(self, anim_args, seed=-1, *args, **kwargs):


        self.fi = FrameInterpolator(anim_args.max_frames, seed)
        self.angle_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.angle))
        self.transform_center_x_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.transform_center_x))
        self.transform_center_y_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.transform_center_y))
        self.zoom_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.zoom))
        self.translation_x_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.translation_x))
        self.translation_y_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.translation_y))
        self.translation_z_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.translation_z))
        self.rotation_3d_x_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.rotation_3d_x))
        self.rotation_3d_y_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.rotation_3d_y))
        self.rotation_3d_z_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.rotation_3d_z))
        self.perspective_flip_theta_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.perspective_flip_theta))
        self.perspective_flip_phi_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.perspective_flip_phi))
        self.perspective_flip_gamma_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.perspective_flip_gamma))
        self.perspective_flip_fv_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.perspective_flip_fv))
        self.noise_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.noise_schedule))
        self.strength_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.strength_schedule))
        self.contrast_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.contrast_schedule))
        self.cfg_scale_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.cfg_scale_schedule))
        self.ddim_eta_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.ddim_eta_schedule))
        self.ancestral_eta_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.ancestral_eta_schedule))
        self.pix2pix_img_cfg_scale_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.pix2pix_img_cfg_scale_schedule))
        self.subseed_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.subseed_schedule))
        self.subseed_strength_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.subseed_strength_schedule))
        self.checkpoint_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.checkpoint_schedule), is_single_string=True)
        self.steps_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.steps_schedule))



        self.seed_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.seed_schedule))
        self.sampler_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.sampler_schedule),
                                                              is_single_string=True)
        self.clipskip_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.clipskip_schedule))
        self.noise_multiplier_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.noise_multiplier_schedule))
        self.mask_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.mask_schedule),
                                                           is_single_string=True)
        self.noise_mask_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.noise_mask_schedule), is_single_string=True)
        self.kernel_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.kernel_schedule))
        self.sigma_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.sigma_schedule))
        self.amount_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.amount_schedule))
        self.threshold_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.threshold_schedule))
        self.aspect_ratio_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.aspect_ratio_schedule))
        self.fov_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.fov_schedule))
        self.near_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.near_schedule))
        self.cadence_flow_factor_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.cadence_flow_factor_schedule))
        self.redo_flow_factor_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.redo_flow_factor_schedule))
        self.far_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.far_schedule))
        self.hybrid_comp_alpha_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.hybrid_comp_alpha_schedule))
        self.hybrid_comp_mask_blend_alpha_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.hybrid_comp_mask_blend_alpha_schedule))
        self.hybrid_comp_mask_contrast_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.hybrid_comp_mask_contrast_schedule))
        self.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.hybrid_comp_mask_auto_contrast_cutoff_high_schedule))
        self.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.hybrid_comp_mask_auto_contrast_cutoff_low_schedule))
        self.hybrid_flow_factor_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(anim_args.hybrid_flow_factor_schedule))


class ControlNetKeys:
    def __init__(self, anim_args, controlnet_args):
        self.fi = FrameInterpolator(max_frames=anim_args.max_frames)
        self.schedules = {}
        for i in range(1, 6):  # 5 CN models in total
            for suffix in ['weight', 'guidance_start', 'guidance_end']:
                prefix = f"cn_{i}"
                key = f"{prefix}_{suffix}_schedule_series"
                self.schedules[key] = self.fi.get_inbetweens(
                    self.fi.parse_key_frames(getattr(controlnet_args, f"{prefix}_{suffix}")))
                setattr(self, key, self.schedules[key])


class LooperAnimKeys:
    def __init__(self, loop_args, anim_args, seed):
        self.fi = FrameInterpolator(anim_args.max_frames, seed)
        self.use_looper = loop_args.use_looper
        self.imagesToKeyframe = loop_args.init_images
        self.image_strength_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(loop_args.image_strength_schedule))
        self.blendFactorMax_series = self.fi.get_inbetweens(self.fi.parse_key_frames(loop_args.blendFactorMax))
        self.blendFactorSlope_series = self.fi.get_inbetweens(self.fi.parse_key_frames(loop_args.blendFactorSlope))
        self.tweening_frames_schedule_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(loop_args.tweening_frames_schedule))
        self.color_correction_factor_series = self.fi.get_inbetweens(
            self.fi.parse_key_frames(loop_args.color_correction_factor))


class FrameInterpolator:
    def __init__(self, max_frames=0, seed=-1) -> None:
        self.max_frames = max_frames
        self.seed = seed

    def sanitize_value(self, value):
        return value.replace("'", "").replace('"', "").replace('(', "").replace(')', "")

    def get_inbetweens(self, key_frames, integer=False, interp_method='Linear', is_single_string=False):
        key_frame_series = pd.Series([np.nan for a in range(self.max_frames)])
        # get our ui variables set for numexpr.evaluate
        global max_f
        global s
        max_f = self.max_frames - 1
        s = self.seed
        value_is_number = None
        value = None
        for i in range(0, self.max_frames):
            if i in key_frames:
                value = key_frames[i]



                value_is_number = check_is_number(self.sanitize_value(value))
                if value_is_number:  # if it's only a number, leave the rest for the default interpolation
                    key_frame_series[i] = self.sanitize_value(value)
            if not value_is_number and value is not None:
                global t
                t = i
                # workaround for values formatted like 0:("I am test") //used for sampler schedules
                key_frame_series[i] = numexpr.evaluate(str(value), casting='unsafe') if not is_single_string else self.sanitize_value(value)
            elif is_single_string:  # take previous string value and replicate it
                key_frame_series[i] = key_frame_series[i - 1]
        key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series  # as string

        if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
            interp_method = 'Quadratic'
        if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
            interp_method = 'Linear'

        key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
        key_frame_series[self.max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
        key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
        if integer:
            return key_frame_series.astype(int)
        return key_frame_series

    def parse_key_frames(self, string):
        # because math functions (i.e. sin(t)) can utilize brackets
        # it extracts the value in form of some stuff
        # which has previously been enclosed with brackets and
        # with a comma or end of line existing after the closing one
        global max_f, s
        frames = dict()
        if string is None:
            string = ""
        for match_object in string.split(","):
            frameParam = match_object.split(":")
            max_f = self.max_frames - 1
            s = self.seed
            frame = int(self.sanitize_value(frameParam[0])) if check_is_number(
                self.sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(
                frameParam[0].strip().replace("'", "", 1).replace('"', "", 1)[::-1].replace("'", "", 1).replace('"', "",
                                                                                                                1)[
                ::-1]))
            frames[frame] = frameParam[1].strip()
        if frames == {} and len(string) != 0:
            raise RuntimeError('Key Frame string not correctly formatted')
        return frames
