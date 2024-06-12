import argparse
import os
import random
import shutil
import sys
import tempfile
from typing import Sequence, Mapping, Any, Union
import json

import cv2
import requests
import torch

from pydub import AudioSegment
import io

from deforum import logger
from deforum.utils.constants import config


def lazy_eval(func):
    class Cache:
        def __init__(self, func):
            self.res = None
            self.func = func
        def get(self):
            if self.res is None:
                self.res = self.func()
            return self.res
    cache = Cache(func)
    return lambda : cache.get()
def get_audio(file, start_time=0, duration=0):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file)

        # Convert start_time and duration from seconds to milliseconds
        start_time_ms = start_time * 1000
        duration_ms = duration * 1000

        # Slice the audio segment if start_time or duration is specified
        if start_time > 0 or duration > 0:
            end_time_ms = start_time_ms + duration_ms if duration > 0 else None
            audio = audio[start_time_ms:end_time_ms]

        # Export the audio segment to a byte stream
        audio_bytes = io.BytesIO()
        audio.export(audio_bytes, format="wav")
        audio_bytes.seek(0)

        # Return the byte content
        return audio_bytes.read()
    except Exception as e:
        logger.warning(f"Failed to extract audio from: {file}")
        return False
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path(comfy_path) -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    # comfyui_path = find_path("ComfyUI")
    if comfy_path is not None and os.path.isdir(comfy_path):
        sys.path.append(comfy_path)
        print(f"'{comfy_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
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




class AnimateRad:
    @torch.inference_mode()
    def __init__(self, **kwargs):
        import_custom_nodes()

        from nodes import (
            CLIPVisionLoader,
            CLIPTextEncode,
            CheckpointLoaderSimple,
            ImageScale,
            ControlNetApplyAdvanced,
            VAELoader,
            VAEEncode,
            LoadImage,
            NODE_CLASS_MAPPINGS,
            LoraLoaderModelOnly,
        )

        self.vae_loader = VAELoader()
        self.video_loader = NODE_CLASS_MAPPINGS["VHS_LoadVideoPath"]()
        self.image_rescaler = ImageScale()
        self.vae_encoder = VAEEncode()
        self.controlnet_loader = NODE_CLASS_MAPPINGS["ControlNetLoaderAdvanced"]()
        self.chekpoint_loader = CheckpointLoaderSimple()
        self.ipadapter_loader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
        self.clipvision_loader = CLIPVisionLoader()
        self.adiff_context_option_creator = NODE_CLASS_MAPPINGS[
            "ADE_AnimateDiffUniformContextOptions"
        ]()
        self.image_loader = LoadImage()
        self.cliptextencoder = CLIPTextEncode()
        self.lora_loader_model_only = LoraLoaderModelOnly()
        self.hiresfix_script_creator = NODE_CLASS_MAPPINGS["HighRes-Fix Script"]()
        self.midas_preprocessor = NODE_CLASS_MAPPINGS["MiDaS-DepthMapPreprocessor"]()
        self.controlnet_apply = ControlNetApplyAdvanced()
        self.prep_image_for_clipvision = NODE_CLASS_MAPPINGS["PrepImageForClipVision"]()
        self.ipadapter_noise = NODE_CLASS_MAPPINGS["IPAdapterNoise"]()
        self.ipadapter_batch = NODE_CLASS_MAPPINGS["IPAdapterBatch"]()
        self.ipadapter_apply_advanced = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
        self.adiff_loader = NODE_CLASS_MAPPINGS[
            "ADE_AnimateDiffLoaderWithContext"
        ]()
        self.ksampler_adv = NODE_CLASS_MAPPINGS["KSampler Adv. (Efficient)"]()
        self.rife_vfi = NODE_CLASS_MAPPINGS["RIFE VFI"]()
        self.video_save = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        self.load_models(sd=kwargs.get("sd_model", "epicrealism_naturalSin.safetensors"),
                         lora=kwargs.get("lora", "AnimateLCM_sd15_t2v_lora.safetensors"))

    def load_models(self, sd, lora):
        self.vae = self.vae_loader.load_vae(vae_name="vae-ft-mse-840000-ema-pruned.safetensors")
        self.controlnet_depth = self.controlnet_loader.load_controlnet(
            control_net_name="control_v11f1p_sd15_depth.pth"
        )
        self.loaded_sd = self.chekpoint_loader.load_checkpoint(
            ckpt_name=sd
        )
        self.loaded_clipvision = self.clipvision_loader.load_clip(
            clip_name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
        )
        self.loaded_sd_animatelcm = self.lora_loader_model_only.load_lora_model_only(
            lora_name=lora,
            strength_model=1,
            model=get_value_at_index(self.loaded_sd, 0),
        )
        self.loaded_ipadapter = self.ipadapter_loader.load_ipadapter_model(
            ipadapter_file="ip-adapter-plus_sd15.safetensors")

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        """
        Executes the animation pipeline.

        Args:
            video_path (str): Path to the input video file. If a URL is provided, the video will be downloaded.
            audio_path (str): Path to the input audio file. If a URL is provided, the video will be downloaded.
            ip_image (str, optional): Path or URL to the input image. If not provided, the first frame of the video will be used.
            max_frames (int, optional): Maximum number of frames to load from the video. Defaults to 0 (no limit).
            use_every_nth (int, optional): Use every nth frame from the video. Defaults to 1.
            width (int, optional): Width of the output video frames. Defaults to 768.
            height (int, optional): Height of the output video frames. Defaults to 768.
            closed_loop (bool, optional): Whether to use closed loop for context options. Defaults to False.
            prompt (str, optional): Prompt for positive conditioning. Defaults to "beautiful flowers, eyeballs, dark colors, mushrooms (centered, symmetric), in the style of clay art".
            negative_prompt (str, optional): Prompt for negative conditioning. Defaults to "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck".
            hires_steps (int, optional): Number of steps for high-resolution pass. Defaults to 6.
            hires_pass_denoise (float, optional): Denoise strength for high-resolution pass. Defaults to 0.56.
            controlnet_strength (float, optional): Strength of the ControlNet conditioning. Defaults to 1.0.
            controlnet_start (float, optional): Start percentage of ControlNet conditioning. Defaults to 0.
            controlnet_end (float, optional): End percentage of ControlNet conditioning. Defaults to 0.5.
            ip_adapter_video_strength (float, optional): Strength of the IP Adapter conditioning for the video. Defaults to 0.85.
            ip_adapter_video_start (float, optional): Start percentage of IP Adapter conditioning for the video. Defaults to 0.
            ip_adapter_video_end (float, optional): End percentage of IP Adapter conditioning for the video. Defaults to 0.8.
            ip_adapter_image_strength (float, optional): Strength of the IP Adapter conditioning for the image. Defaults to 0.85.
            ip_adapter_image_start (float, optional): Start percentage of IP Adapter conditioning for the image. Defaults to 0.
            ip_adapter_image_end (float, optional): End percentage of IP Adapter conditioning for the image. Defaults to 0.5.
            seed (int, optional): Random seed for sampling. Defaults to -1 (random seed).
            steps (int, optional): Number of steps for sampling. Defaults to 6.
            cfg (float, optional): Classifier-free guidance scale. Defaults to 1.3.
            sampler (str, optional): Sampler type. Defaults to "lcm".
            scheduler (str, optional): Scheduler type. Defaults to "sgm_uniform".
            start_at_step (int, optional): Start step for sampling. Defaults to 2.
            fps (int, optional): Frame rate of the output video. Defaults to 24.

        Returns:
            Any: The saved video.
        """

        video_path = kwargs.get('video_path')
        audio_path = kwargs.get('audio_path')
        ip_image = kwargs.get('ip_image')
        seed = kwargs.get('seed', -1)

        if seed == -1:
            seed = random.randint(1, 2 ** 64)
        assert video_path is not None, "Video path must be provided"

        # Check if video_path is a URL and download if necessary
        if video_path.startswith("http://") or video_path.startswith("https://"):
            response = requests.get(video_path, stream=True)
            temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            with open(temp_video_file.name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            video_path = temp_video_file.name

        # loaded_ip_image = self.image_loader.load_image(image="input.png")
        loaded_video = self.video_loader.load_video(
            video=video_path,
            frame_load_cap=kwargs.get('max_frames', 0),
            force_rate=0,
            force_size='Disabled',
            skip_first_frames=0,
            select_every_nth=kwargs.get('use_every_nth', 1),
            custom_width=None,
            custom_height=None)

        # Process ip_image: download if URL, copy if local file
        ip_image_path = os.path.join(config.comfy_path, "input/ip_image.png")
        os.makedirs(os.path.join(config.comfy_path, "input"), exist_ok=True)

        if ip_image is None:
            # If ip_image is None, extract the first frame of the video and save it to input/ip_image.png
            cap = cv2.VideoCapture(video_path)
            success, frame = cap.read()
            if success:
                cv2.imwrite(ip_image_path, frame)
            cap.release()
        else:
            if ip_image.startswith("http://") or ip_image.startswith("https://"):
                response = requests.get(ip_image, stream=True)
                with open(ip_image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                shutil.copy(ip_image, ip_image_path)

        loaded_ip_image = self.image_loader.load_image(image='ip_image.png')

        scaled_init_video = self.image_rescaler.upscale(
            upscale_method="nearest-exact",
            width=kwargs.get('width', 768),
            height=kwargs.get('height', 768),
            crop="center",
            image=get_value_at_index(loaded_video, 0),
        )
        encoded_video = self.vae_encoder.encode(
            pixels=get_value_at_index(scaled_init_video, 0),
            vae=get_value_at_index(self.vae, 0),
        )
        adiff_context_options = (
            self.adiff_context_option_creator.create_options(
                context_schedule="uniform", fuse_method="flat",
                context_length=16, context_stride=1,
                context_overlap=4, closed_loop=kwargs.get('closed_loop', False)
            )
        )
        positive_conditioning = self.cliptextencoder.encode(
            text=kwargs.get('prompt',
                            "beautiful flowers, eyeballs, dark colors, mushrooms (centered, symmetric), in the style of clay art"),
            clip=get_value_at_index(self.loaded_sd, 1),
        )

        negative_conditioning = self.cliptextencoder.encode(
            text=kwargs.get('negative_prompt',
                            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"),
            clip=get_value_at_index(self.loaded_sd, 1),
        )

        hiresfix_script = self.hiresfix_script_creator.hires_fix_script(
            upscale_type="latent",
            hires_ckpt_name="(use same)",
            latent_upscaler="nearest-exact",
            pixel_upscaler="4x-UltraSharp.pth",
            upscale_by=1.25,
            use_same_seed=True,
            seed=seed,
            hires_steps=kwargs.get('hires_steps', 6),
            denoise=kwargs.get('hires_pass_denoise', 0.56),
            iterations=1,
            use_controlnet=False,
            control_net_name="TTPLANET_Controlnet_Tile_realistic_v1_fp32.safetensors",
            strength=1,
            preprocessor="none",
            preprocessor_imgs=False,
        )
        if kwargs.get('controlnet_strength', 1.0) > 0.0:

            loaded_video_depth_maps = self.midas_preprocessor.execute(
                a=6.283185307179586,
                bg_threshold=0.1,
                resolution=512,
                image=get_value_at_index(loaded_video, 0),
            )

            applied_controlnet = self.controlnet_apply.apply_controlnet(
                strength=kwargs.get('controlnet_strength', 1.0),
                start_percent=kwargs.get('controlnet_start', 0),
                end_percent=kwargs.get('controlnet_end', 0.5),
                positive=get_value_at_index(positive_conditioning, 0),
                negative=get_value_at_index(negative_conditioning, 0),
                control_net=get_value_at_index(self.controlnet_depth, 0),
                image=get_value_at_index(loaded_video_depth_maps, 0),
            )

        loaded_video_for_clipvision = self.prep_image_for_clipvision.prep_image(
            interpolation="LANCZOS",
            crop_position="pad",
            sharpening=0,
            image=get_value_at_index(loaded_video, 0),
        )

        negative_clipvision_input = self.ipadapter_noise.make_noise(
            type="shuffle",
            strength=0.5,
            blur=0,
            image_optional=get_value_at_index(loaded_video_for_clipvision, 0),
        )

        loaded_sd_applied_ipadapter_video = self.ipadapter_batch.apply_ipadapter(
            weight=kwargs.get('ip_adapter_video_strength', 0.85),
            weight_type="linear",
            start_at=kwargs.get('ip_adapter_video_start', 0),
            end_at=kwargs.get('ip_adapter_video_end', 0.8),
            embeds_scaling="K+V w/ C penalty",
            model=get_value_at_index(self.loaded_sd_animatelcm, 0),
            ipadapter=get_value_at_index(self.loaded_ipadapter, 0),
            image=get_value_at_index(loaded_video_for_clipvision, 0),
            image_negative=get_value_at_index(negative_clipvision_input, 0),
            clip_vision=get_value_at_index(self.loaded_clipvision, 0),
        )

        loaded_ip_image_clipvision = self.prep_image_for_clipvision.prep_image(
            interpolation="LANCZOS",
            crop_position="pad",
            sharpening=0,
            image=get_value_at_index(loaded_ip_image, 0),
        )

        ip_image_negative = self.ipadapter_noise.make_noise(
            type="shuffle",
            strength=0.5,
            blur=0,
            image_optional=get_value_at_index(loaded_ip_image_clipvision, 0),
        )

        loaded_sd_applied_ipadapter_video_and_image = self.ipadapter_apply_advanced.apply_ipadapter(
            weight=kwargs.get('ip_adapter_image_strength', 0.85),
            weight_type="linear",
            combine_embeds="concat",
            start_at=kwargs.get('ip_adapter_image_start', 0),
            end_at=kwargs.get('ip_adapter_image_end', 0.5),
            embeds_scaling="K+V w/ C penalty",
            model=get_value_at_index(loaded_sd_applied_ipadapter_video, 0),
            ipadapter=get_value_at_index(self.loaded_ipadapter, 0),
            image=get_value_at_index(loaded_ip_image_clipvision, 0),
            image_negative=get_value_at_index(ip_image_negative, 0),
            clip_vision=get_value_at_index(self.loaded_clipvision, 0),
        )

        sd_with_ip_animatediff_model = (
            self.adiff_loader.load_mm_and_inject_params(
                model_name=kwargs.get("ad_model", "AnimateLCM_sd15_t2v.ckpt"),
                beta_schedule=kwargs.get("beta_schedule", "lcm >> sqrt_linear"),
                motion_scale=1,
                apply_v2_models_properly=False,
                model=get_value_at_index(loaded_sd_applied_ipadapter_video_and_image, 0),
                context_options=get_value_at_index(
                    adiff_context_options, 0
                ),
            )
        )

        if kwargs.get('hires_steps', 0) > 0:
            hires = get_value_at_index(hiresfix_script, 0)
        else:
            hires = None
        if kwargs.get('controlnet_strength', 1.0) > 0.0:
            positive = get_value_at_index(applied_controlnet, 0)
            negative = get_value_at_index(applied_controlnet, 1)
        else:
            positive = get_value_at_index(positive_conditioning, 0)
            negative = get_value_at_index(negative_conditioning, 0)

        print("AD SEED", seed)

        result_samples = self.ksampler_adv.sample_adv(
            add_noise="enable",
            noise_seed=seed,
            steps=kwargs.get('steps', 6),
            cfg=kwargs.get('cfg', 1.3),
            sampler_name=kwargs.get('sampler', "lcm"),
            scheduler=kwargs.get('scheduler', "sgm_uniform"),
            start_at_step=kwargs.get('start_at_step', 2),
            end_at_step=10000,
            return_with_leftover_noise="disable",
            preview_method="auto",
            vae_decode="true",
            model=get_value_at_index(sd_with_ip_animatediff_model, 0),
            positive=positive,
            negative=negative,
            latent_image=get_value_at_index(encoded_video, 0),
            optional_vae=get_value_at_index(self.vae, 0),
            script=hires,
        )

        interpolated_samples = self.rife_vfi.vfi(
            ckpt_name="rife47.pth",
            clear_cache_after_n_frames=10,
            multiplier=2,
            fast_mode=True,
            ensemble=True,
            scale_factor=1,
            frames=get_value_at_index(result_samples, 5),
        )
        if audio_path is not None:
            a_func = lambda : get_audio(audio_path)
            a = lazy_eval(a_func)
        else:
            a = get_value_at_index(loaded_video, 2)

        saved_video = self.video_save.combine_video(
            frame_rate=kwargs.get('fps', 24) * 2,
            loop_count=0,
            filename_prefix="deforumDiff",
            format="video/h264-mp4",
            pingpong=False,
            save_output=True,
            images=get_value_at_index(interpolated_samples, 0),
            audio=a,
            unique_id=14314448330963208959,
        )

        from comfy.model_management import cleanup_models, unload_all_models
        unload_all_models()
        cleanup_models()

        return saved_video




def run_pipeline_with_config(config_path: str):
    """
    Loads a configuration from a JSON file and uses it as kwargs to run the AnimateRad animation pipeline.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        Any: The result of the pipeline execution.
    """
    # Load the configuration from the JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Run the pipeline with the loaded configuration
    result = pipeline(**config)

    return result['result'][0]
comfy_path = ""

# Example usage with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AnimateRad pipeline with a configuration file.")
    parser.add_argument("config_path", type=str, help="Path to the JSON configuration file")
    parser.add_argument("comfy_path", type=str, help="Path to the JSON configuration file")
    args = parser.parse_args()

    # Keep only the config_path in sys.argv and clear the rest
    sys.argv = [sys.argv[0]]
    add_comfyui_directory_to_sys_path(args.comfy_path)
    add_extra_model_paths()
    comfy_path = args.comfy_path


    pipeline = AnimateRad()
    result = run_pipeline_with_config(args.config_path)
