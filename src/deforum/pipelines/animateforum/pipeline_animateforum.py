

"""

instantiate DeforumAnimationPipeline
instantiate DeforumAnimateDiffPipeline

__call__

add animatediff_handler function to shoot_fns

setup deforum

animatediff_handler_function


    generate sequence on max_frames % animatediff_max_frames == 0

    else:

        warp = warp_keys [last_animatediff_generation - frame_idx]

        set prev_img
        set next_img (for cadence)
        set hybrid_motion_ somehow...

"""
import gc
import time
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..deforum_animation.animation_helpers import anim_frame_warp_3d_direct
from ..deforum_animation.pipeline_deforum_animation import (DeforumAnimationPipeline)
from ...generators.deforum_flow_generator import get_reliable_flow_from_images

from ...pipeline_utils import DeforumGenerationObject
from ...utils.constants import config
from ...utils.deforum_logger_util import Logger
from deforum.utils.logging_config import logger

import numpy as np
import cv2

from ...utils.image_utils import image_transform_optical_flow


def blend_pixels(pixel1, pixel2, mode):
    if mode == 'multiply':
        return pixel1 * pixel2 / 255
    elif mode == 'screen':
        return 255 - ((255 - pixel1) * (255 - pixel2) / 255)
    elif mode == 'overlay':
        return pixel2 * (pixel1 / 255) if pixel1 < 128 else 255 - (255 - pixel2) * (255 - pixel1) / 255
    elif mode == 'darken':
        return np.minimum(pixel1, pixel2)
    elif mode == 'lighten':
        return np.maximum(pixel1, pixel2)
    elif mode == 'dodge':
        return np.minimum(255, pixel1 * 255 / (255 - pixel2))
    elif mode == 'burn':
        return 255 - np.minimum(255, (255 - pixel1) * 255 / pixel2)
    elif mode == 'soft_light':
        return ((1 - (pixel1 / 255)) * pixel2) + ((pixel1 / 255) * (255 - (255 - pixel2) * (255 - pixel1) / 255))
    elif mode == 'hard_light':
        return blend_pixels(pixel2, pixel1, 'overlay')
    else:
        # 'normal' and any other unspecified modes
        return pixel2

def enhanced_pixel_diffusion_blend(image1, image2, alpha=0.5, beta=0.5, noise='uniform', edge_preserve=False, gradient=False, blend_mode='normal'):
    if alpha + beta > 1:
        raise ValueError("Alpha + Beta should not exceed 1.")

    if noise not in ['uniform', 'gaussian']:
        raise ValueError("Noise type must be 'uniform' or 'gaussian'.")

    height, width, _ = image1.shape
    result = np.copy(image1)

    # Generate noise maps for alpha and beta blending if needed
    if noise == 'uniform':
        alpha_noise = np.random.uniform(0, 1, (height, width))
        beta_noise = np.random.uniform(0, 1, (height, width))
    elif noise == 'gaussian':
        alpha_noise = np.abs(np.random.randn(height, width))
        alpha_noise /= alpha_noise.max()  # Normalize to 0-1
        beta_noise = np.abs(np.random.randn(height, width))
        beta_noise /= beta_noise.max()  # Normalize to 0-1

    # Apply edge detection if requested
    if edge_preserve:
        edges = cv2.Canny(image1, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) / 255  # Normalize and convert to 3 channel

    for row in range(height):
        for col in range(width):
            # Adjust alpha and beta if gradient is enabled
            if gradient:
                alpha_grad = alpha * (col / width)
                beta_grad = beta * ((width - col) / width)
            else:
                alpha_grad = alpha
                beta_grad = beta

            # Decide which pixel to take based on noise and edge preservation
            take_from_image2 = alpha_noise[row, col] < alpha_grad
            take_from_image1 = beta_noise[row, col] < beta_grad
            is_edge = edge_preserve and edges[row, col].any()

            if take_from_image2 and not is_edge:
                selected_pixel = image2[row, col]
            elif take_from_image1 and not is_edge:
                selected_pixel = image1[row, col]
            else:
                selected_pixel = result[row, col]  # Default or edge

            # Apply the selected blend mode
            result[row, col] = blend_pixels(result[row, col], selected_pixel, blend_mode)

    return result


def calculate_blurriness_metric(image_np_array):
    """
    Calculates the blurriness metric of an image.

    :param image_np_array: np.array of the image
    :return: Blurriness metric, where 1.0 is very blurry and 0.0 is sharp
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian operator to the grayscale image
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    # Invert the normalized variance to get the blurriness metric
    # This is a simple transformation and may need to be adjusted based on the expected range of variances
    blurriness_metric = 1 / (1 + laplacian_var)

    # Make sure the metric is between 0 and 1
    blurriness_metric = np.clip(blurriness_metric, 0, 1)

    return blurriness_metric
def custom_pixel_diffusion_blend(image1, alpha, image2, beta):
    if alpha + beta > 1:
        raise ValueError("Alpha + Beta should not exceed 1.")

    height, width, channels = image1.shape
    result = np.copy(image1)

    for row in range(height):
        for col in range(width):
            if np.random.uniform(0, 1) < alpha:
                result[row, col] = image2[row, col]
            elif np.random.uniform(0, 1) < beta:
                result[row, col] = image1[row, col]

    return result

def animatediff_handler(cls):
    if not hasattr(cls.gen, "animatediff_max_frames"):
        cls.gen.animatediff_max_frames = cls.gen.max_frames
        #cls.gen.animatediff_max_frames = 16
    if not hasattr(cls.gen, "animatediff_last_gen"):
        cls.gen.animatediff_last_gen = 0
    if not hasattr(cls.gen, "animatediff_steps"):
        cls.gen.animatediff_steps = 25
    if not hasattr(cls, "animate_flow"):
        cls.animate_flow = None
    logger.info(f"ANIMATION HANDLER")
    #DETERMINES STRENGTH OF THE ORIGINAL IMAGE vs ANIMATEDIFF INPUT
    alpha = 0.55
    beta = 1.0 - alpha

    if ((cls.gen.frame_idx + 1) - cls.gen.animatediff_last_gen) % cls.gen.animatediff_max_frames == 0 or cls.gen.frame_idx == 0:# or (cls.gen.frame_idx - cls.gen.animatediff_last_gen) > cls.gen.animatediff_max_frames + 1:
        # if cls.gen.prev_img is not None:
        #
        #     to_encode = cv2.resize(cls.gen.prev_img,
        #                                  (int(cls.gen.prev_img.shape[1] // 2), int(cls.gen.prev_img.shape[0] // 2)))
        #
        #     latent = torch.from_numpy(to_encode.astype(np.float32) / 255.0).unsqueeze(0)
        #     latents = [cls.generator.encode_latent(latent, cls.gen.seed, cls.gen.subseed, 1.0, 1024, 1024)["samples"]]
        #     for i in range(cls.gen.animatediff_max_frames -1):
        #         l = torch.randn([1, 4, cls.gen.height // 2 // 8, cls.gen.width //2 // 8]).to("cuda")
        #         latents.append(l)
        #     latents = torch.stack(latents, dim=0)
        # else:
        latents = None

        width = int(cls.gen.width // 2)
        height = int(cls.gen.height // 2)
        if width < 512:
            width = 512
        if height < 512:
            height = 512

        anim = cls.animatediff(prompts = cls.gen.prompt_series[cls.gen.frame_idx],
                               max_frames=cls.gen.animatediff_max_frames,
                               frame_interpolation_engine="None",
                               steps= cls.gen.animatediff_steps,
                               width = width,
                               height = height,
                               latents=latents,
                               scale=7)
        cls.gen.animatediff_frames = anim.images
        cls.gen.animatediff_last_gen = cls.gen.frame_idx
        #cls.gen.image = Image.fromarray(cls.gen.animatediff_frames[0])
        if cls.gen.frame_idx == 0:
            cls.gen.use_init = True
            cls.gen.init_image = Image.fromarray(cls.gen.animatediff_frames[0]).resize((cls.gen.width, cls.gen.height), resample=Image.Resampling.LANCZOS)
            cls.gen.animatediff_prev_img = cv2.resize(cv2.cvtColor(cls.gen.animatediff_frames[0], cv2.COLOR_RGB2BGR), (cls.gen.width, cls.gen.height))
        if cls.gen.prev_img is not None:
            cls.gen.use_init = False
            cls.gen.init_image = None
            animatediff_img = cv2.cvtColor(cls.gen.animatediff_frames[0], cv2.COLOR_RGB2BGR)
            animatediff_img = cv2.resize(animatediff_img,
                                         (cls.gen.prev_img.shape[1], cls.gen.prev_img.shape[0]))
            cls.gen.opencv_image = cv2.addWeighted(cls.gen.opencv_image, alpha, animatediff_img, beta, 0)
            cls.gen.animatediff_prev_img = animatediff_img
            #cls.gen.image = Image.fromarray(cv2.cvtColor(cls.gen.opencv_image, cv2.COLOR_BGR2RGB))
            cv2.imwrite("current_previmg.png", cls.gen.prev_img)
            cv2.imwrite("current_opencv.png", cls.gen.opencv_image)

    else:
        cls.gen.use_init = False
        cls.gen.init_image = None

        angle = 0
        animatediff_index = cls.gen.frame_idx - cls.gen.animatediff_last_gen
        animatediff_img = cls.gen.animatediff_frames[animatediff_index]
        animatediff_img = cv2.resize(animatediff_img, (cls.gen.opencv_image.shape[1], cls.gen.opencv_image.shape[0]))
        animatediff_img = cv2.cvtColor(animatediff_img, cv2.COLOR_RGB2BGR)
        # animatediff_prev_img = cls.gen.animatediff_frames[animatediff_index - 1]
        # animatediff_prev_img = cv2.resize(animatediff_prev_img, (cls.gen.opencv_image.shape[1], cls.gen.opencv_image.shape[0]))
        # animatediff_prev_img = cv2.cvtColor(animatediff_prev_img, cv2.COLOR_RGB2BGR)

        x = 0.0
        y = 0.0
        z = 0.0
        rx = 0.0
        ry = 0.0
        rz = 0.0
        for key in range(cls.gen.animatediff_last_gen, cls.gen.frame_idx):
            """
            Here we gather and add up all warp values between last_anim_frame - frame_idx
            """
            x += cls.gen.keys.translation_x_series[key]
            y += cls.gen.keys.translation_y_series[key]
            z += cls.gen.keys.translation_z_series[key]
            rx += cls.gen.keys.rotation_3d_x_series[key]
            ry += cls.gen.keys.rotation_3d_y_series[key]
            rz += cls.gen.keys.rotation_3d_z_series[key]
        if any(value > 0.0 for value in [x, y, z, rx, ry, rz]):
            animatediff_img, mask = anim_frame_warp_3d_direct(cls, animatediff_img, x, y, z, rx, ry, rz)



        cls.animate_flow, reliable_flow = get_reliable_flow_from_images(cls.gen.animatediff_prev_img, animatediff_img, cls.gen.hybrid_flow_method, cls.raft_model, cls.animate_flow,
                                                            1.0)
        cls.gen.opencv_image = image_transform_optical_flow(cls.gen.opencv_image, cls.animate_flow,
                                                        cls.gen.hybrid_comp_schedules['flow_factor'])
        cls.gen.prev_img = image_transform_optical_flow(cls.gen.prev_img, cls.animate_flow,
                                                        cls.gen.hybrid_comp_schedules['flow_factor'])


        cv2.imwrite("current_animdiff.png", animatediff_img)
        #

        #
        #
        # # Define the weight of each image
        alpha_opencv = 0.75  # Weight of the first image. The value should be between 0 and 1.
        # alpha_prev = 0.42  # Weight of the first image. The value should be between 0 and 1.
        beta_opencv = 1.0 - alpha_opencv  # Weight of the second image. beta = 1 - alpha
        # beta_prev = 1.0 - alpha_prev
        # # Blend the images
        use_custom_blend = False
        blend_at_all = True
        if blend_at_all:
            if not use_custom_blend:
                cls.gen.prev_img = cv2.addWeighted(cls.gen.prev_img, alpha_opencv, animatediff_img, beta_opencv, 0)
            else:
                cls.gen.prev_img = enhanced_pixel_diffusion_blend(cls.gen.prev_img,
                                                                      animatediff_img,
                                                                      alpha_opencv,
                                                                      beta_opencv,
                                                                      noise="gaussian",
                                                                      edge_preserve=False,
                                                                      gradient=False,
                                                                      blend_mode="normal")
        cv2.imwrite("current_opencv.png", cls.gen.opencv_image)
        cv2.imwrite("current_previmg.png", cls.gen.prev_img)

        #cls.gen.image = Image.fromarray(cv2.cvtColor(cls.gen.opencv_image, cv2.COLOR_BGR2RGB))
        #cls.gen.prev_img = animatediff_prev_img
        cls.gen.animatediff_prev_img = animatediff_img

        logger.info(f"BLURRINESS: {calculate_blurriness_metric(cls.gen.opencv_image)}")
        str = cls.gen.keys.strength_schedule_series[cls.gen.frame_idx]
        adj = calculate_blurriness_metric(cls.gen.prev_img)
        if adj > 0.05:
            cls.gen.keys.strength_schedule_series[cls.gen.frame_idx] = str + (2 * adj)



        """
        Then do the warp on the animatediff_frame to use
        Blend in with cls.gen.image, replace cls.gen.opencv_image
        """
    return

class DeforumAnimateDifforumPipeline(DeforumAnimationPipeline):

    def __init__(self, generator: Callable, logger: Optional[Callable] = None, *args, **kwargs):
        """
        Initialize the DeforumAnimationPipeline.

        Args:
            generator (Callable): The generator function for producing animations.
            logger (Optional[Callable], optional): Optional logger function. Defaults to None.
        """
        super().__init__(generator, logger)
        from ..animatediff_animation.pipeline_animatediff_animation import DeforumAnimateDiffPipeline

        animatediff_model_id = kwargs.pop("animatediff_model_id", "132632")

        self.animatediff = DeforumAnimateDiffPipeline.from_civitai(model_id=animatediff_model_id)


    def reset(self, *args, **kwargs) -> None:
        super().reset()
        self.shoot_fns.insert(0, animatediff_handler)
