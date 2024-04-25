import os

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLPipeline, \
    StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

from .rng_noise_generator import ImageRNGNoise
from ..utils.logging_config import logger


class DeforumDiffusersGenerator:
    def __init__(self, model_path: str = None, skip_load=False, *args, **kwargs):
        if not skip_load:
            if model_path:
                if self.is_url(model_path):
                    if 'huggingface.co' in model_path:
                        print("Loading from Hugging Face URL...")
                        self.load_from_huggingface(model_path)
                    else:
                        print("Loading from a general URL...")
                        self.load_from_url(model_path)
                elif os.path.exists(model_path):
                    print("Loading from a local file path...")
                    self.load_from_local(model_path)
                else:
                    raise ValueError("Provided model path does not exist and does not appear to be a valid URL")
            else:
                raise ValueError("No model path provided, and skip_load is False")

    def is_url(self, path: str) -> bool:
        """ Check if the given path is a URL """
        return path.startswith('http://') or path.startswith('https://')

    def load_from_huggingface(self, url: str):
        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(url,
                                                  use_safetensors=True,
                                                  torch_dtype=torch.float16).to('cuda')
            self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
            logger.info(f"Loaded safetensors from Huggingface, {url}.")
        except:
            logger.info("Error loading safetensors from Huggingface, trying bin's.")
            self.pipe = AutoPipelineForText2Image(url,
                                                  use_safetensors=False,
                                                  torch_dtype=torch.float16)
            self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
    def load_from_url(self, url: str):
        # Logic to load a model from a general URL
        pass

    def load_from_local(self, file_path: str, version='sd_xl'):

        if version == 'sd_xl':
            pipe_class = StableDiffusionXLPipeline
        elif version == 'sd_classic':
            pipe_class = StableDiffusionPipeline
        # Logic to load a model from a local file system
        try:
            self.pipe = pipe_class.from_single_file(file_path,
                                                  torch_dtype=torch.float16).to('cuda')
            self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
            logger.info(f"Loaded model from local file: {file_path}.")
        except:
            logger.info("Error loading from single file.")

    def generate_latent(self, width, height, seed, subseed, subseed_strength, seed_resize_from_h=None,
                        seed_resize_from_w=None, reset_noise=False):
        shape = [4, height // 8, width // 8]
        if self.rng is None or reset_noise:
            self.rng = ImageRNGNoise(shape=shape, seeds=[seed], subseeds=[subseed], subseed_strength=subseed_strength,
                                     seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
        noise = self.rng.next()
        # noise = torch.zeros([1, 4, width // 8, height // 8])
        return {"samples": noise}

    def __call__(self,
                 prompt="",
                 pooled_prompts=None,
                 next_prompt=None,
                 prompt_blend=None,
                 negative_prompt="",
                 steps=25,
                 scale=7.5,
                 sampler_name="dpmpp_2m_sde",
                 scheduler="karras",
                 width=None,
                 height=None,
                 seed=-1,
                 strength=1.0,
                 init_image=None,
                 subseed=-1,
                 subseed_strength=0.6,
                 cnet_image=None,
                 cond=None,
                 n_cond=None,
                 return_latent=None,
                 latent=None,
                 last_step=None,
                 seed_resize_from_h=1024,
                 seed_resize_from_w=1024,
                 reset_noise=False,
                 enable_prompt_blend=True,
                 use_areas= False,
                 areas= None,
                 *args,
                 **kwargs):
        torch.manual_seed(seed)
        if init_image is None:
            image = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=scale,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                output_type="pil",
            ).images[0]
        else:
            image = self.img2img_pipe(
                prompt=prompt,
                strength=strength,
                image=init_image / 255.0,
                width=width,
                height=height,
                guidance_scale=scale,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                output_type="pil",
            ).images[0]
        return image

    def cleanup(self):
        return
