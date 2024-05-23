import os
import secrets

import numpy as np
from PIL import Image
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLPipeline, \
    StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LCMScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import retrieve_timesteps, \
    retrieve_latents
from diffusers.utils.torch_utils import randn_tensor

from .rng_noise_generator import ImageRNGNoise, slerp
from ..utils.logging_config import logger

from diffusers import (
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler
)

try:
    from diffusers import TCDScheduler
except:
    logger.warning("TCD Scheduler not available, please update diffusers")

def configure_scheduler(pipe, model_config):
    if model_config == "DPM++ 2M":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif model_config == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif model_config == "DPM++ 2M SDE":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
    elif model_config == "DPM++ 2M SDE Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif model_config == "DPM++ 2S a":
        # Assuming DPMSolverSinglestepScheduler is very similar to not available (N/A)
        print("Configuration for DPM++ 2S a is not available.")
    elif model_config == "DPM++ 2S a Karras":
        # Similar to the above, assuming a direct correspondence
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif model_config == "DPM++ SDE":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    elif model_config == "DPM++ SDE Karras":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, **pipe.scheduler.config)
    elif model_config == "DPM2":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    elif model_config == "DPM2 Karras":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif model_config == "DPM2 a":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif model_config == "DPM2 a Karras":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif model_config == "DPM adaptive" or model_config == "DPM fast":
        # Assuming these configurations are not available as no specific scheduler is indicated
        print("Configuration for DPM adaptive or DPM fast is not available.")
    elif model_config == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif model_config == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif model_config == "Heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif model_config == "LMS":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif model_config == "LMS Karras":
        pipe.scheduler = LMSDiscreteScheduler.from_config( pipe.scheduler.config, use_karras_sigmas=True)
    elif model_config == "Deis":
        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    elif model_config == "UniPcMultiStep":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    elif model_config == "LCM":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif model_config == "TCD":
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    else:
        logger.info("Unsupported scheduler, not switching")

# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
# def slerp(val, low, high):
#     """
#     Perform SLERP between two points in the latent space for tensors of shape [1, 4, h, w].
#     Args:
#     - val (float): Interpolation value between 0 and 1.
#     - low, high (torch.Tensor): Tensors representing the starting and ending points of the interpolation.
#     """
#     # Flatten the spatial dimensions to treat each channel as a separate vector
#     low_flat = low.flatten(start_dim=2)  # Shape: [1, 4, h*w]
#     high_flat = high.flatten(start_dim=2)  # Shape: [1, 4, h*w]
#
#     # Compute dot product and norms for cosine of angle
#     dot_product = torch.sum(low_flat * high_flat, dim=2)  # Sum along the spatial dimension
#     norms = torch.norm(low_flat, dim=2) * torch.norm(high_flat, dim=2)
#
#     # Compute the angle omega using arccosine safely
#     omega = torch.acos(torch.clamp(dot_product / norms, -1, 1))
#
#     # Compute the sine of omega, and handle the special case of very small omega
#     sin_omega = torch.sin(omega)
#     sin_omega = torch.where(sin_omega == 0, torch.ones_like(sin_omega), sin_omega)  # Avoid division by zero
#
#     # Compute interpolation weights
#     sin_omega_val = torch.sin((1.0 - val) * omega)
#     sin_val_omega = torch.sin(val * omega)
#
#     # Apply spherical interpolation
#     result_flat = (sin_omega_val / sin_omega).unsqueeze(2) * low_flat + (sin_val_omega / sin_omega).unsqueeze(2) * high_flat
#
#     # Reshape back to original shape [1, 4, h, w]
#     result = result_flat.view_as(low)
#
#     return result

class DeforumDiffusersGenerator:
    def __init__(self, model_path: str = None, skip_load=False, *args, **kwargs):
        if not skip_load:
            self.load_pipeline(model_path)
        self.selected_scheduler = ""
        self.model_path = model_path
        self.rng = None

    def optimize(self):
        try:
            from sfast.compilers.diffusion_pipeline_compiler import (
                compile,
                CompilationConfig,
            )

            self.pipe.config.force_upcast = False
            self.pipe.watermarker = None
            self.pipe.safety_checker = None
            self.pipe.set_progress_bar_config(disable=True)

            config = CompilationConfig.Default()
            config.enable_jit = True
            config.enable_jit_freeze = True
            config.enable_cuda_graph = True
            try:
                import triton
                config.enable_triton = True
            except:
                config.enable_triton = False

            config.enable_cnn_optimization = True
            config.preserve_parameters = False
            config.prefer_lowp_gemm = True
            config.enable_xformers = True
            config.channels_last = "channels_last"
            config.enable_fused_linear_geglu = True
            config.trace_scheduler = False

            #_ = self.__call__()
            # self.pipe.vae = torch.compile(self.pipe.vae, mode="reduce-overhead")

            self.pipe = compile(self.pipe, config)
        except:
            pass
    def load_pipeline(self, model_path: str = None):
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
        self.optimize()
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

    def prepare_latents(
        self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
    ):
        return image

    def prepare_latents_with_subseed(
            self, image, num_inference_steps, strength, seed, subseed=None, subseed_strength=0.5
    ):
        device = self.img2img_pipe._execution_device

        self.img2img_pipe.prepare_latents = self.prepare_latents
        def denoising_value_valid(dnv):
            return isinstance(dnv, float) and 0 < dnv < 1
        timesteps, num_inference_steps = retrieve_timesteps(self.img2img_pipe.scheduler, num_inference_steps, device, None)
        timesteps, num_inference_steps = self.img2img_pipe.get_timesteps(
            num_inference_steps,
            strength,
            device,
            denoising_start=None,
        )
        latent_timestep = timesteps[:1].repeat(1 * 1)

        add_noise = True


        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        # Offload text encoder if `enable_model_cpu_offload` was enabled
        if hasattr(self.img2img_pipe, "final_offload_hook") and self.img2img_pipe.final_offload_hook is not None:
            self.img2img_pipe.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()
        dtype = torch.float16
        image = image.to(device=device, dtype=dtype)

        batch_size = 1
        generator = torch.Generator(device).manual_seed(seed)
        if image.shape[1] == 4:
            init_latents = image
        else:
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.img2img_pipe.vae.config.force_upcast:
                image = image.float()
                self.img2img_pipe.vae.to(dtype=torch.float32)


            init_latents = retrieve_latents(self.img2img_pipe.vae.encode(image), generator=generator)

            if self.img2img_pipe.vae.config.force_upcast:
                self.img2img_pipe.vae.to(dtype)

            init_latents = init_latents.to(dtype)


        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )

        # if add_noise:
        #     shape = init_latents.shape
        #     noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        #     init_latents = self.img2img_pipe.scheduler.add_noise(init_latents, noise, latent_timestep)



        init_latents = self.img2img_pipe.vae.config.scaling_factor * init_latents
        print(
            f'init_latents min: {torch.min(init_latents).item()}, init_latents max: {torch.max(init_latents).item()}')

        # if subseed is not None:
        #
        #
        #
        #     shape = init_latents.shape
        #     if subseed == -1:
        #         subseed = secrets.randbelow(999999999999999999)
        #     sub_generator = torch.Generator(device=device).manual_seed(subseed)
        #
        #     noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        #     sub_noise = randn_tensor(init_latents.shape, generator=sub_generator, device=device, dtype=dtype)
        #     noise = slerp(subseed_strength, noise, sub_noise)
        #     # get latents
        #     init_latents = self.img2img_pipe.scheduler.add_noise(init_latents, noise, latent_timestep)
        #
        #
        #     logger.info("USING SLERPED LATENTS")
        return init_latents, timesteps

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
        if subseed is not None:
            if subseed == -1:
                subseed = secrets.randbelow(999999999999999999)

        # if self.rng is None or reset_noise:
        shape = [1, 4, height // 8, width // 8]
        self.rng = ImageRNGNoise(shape=shape, seeds=[seed], subseeds=[subseed], subseed_strength=subseed_strength,
                                 seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)


        info = {"strength":str(strength), "steps":str(steps), "seed":str(seed), "subseed":str(subseed), "subseed_strength":str(subseed_strength)}
        args = [f"{key}={value}" for key, value in info.items()]
        logger.info("DEFORUM DIFFUSERS SAMPLER: %s", " ".join(args))
        if not hasattr(self, 'pipe'):
            self.load_pipeline(self.model_path)
        if self.selected_scheduler != sampler_name:
            configure_scheduler(self.pipe, sampler_name)
            configure_scheduler(self.img2img_pipe, sampler_name)
            self.selected_scheduler = sampler_name
        torch.manual_seed(seed)
        if init_image is None or strength > 0.99:
            device = self.img2img_pipe._execution_device
            dtype = torch.float16
            sub_generator = torch.Generator(device=device).manual_seed(subseed)
            generator = torch.Generator(device=device).manual_seed(seed)

            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            sub_noise = randn_tensor(noise.shape, generator=sub_generator, device=device, dtype=dtype)

            print(noise.shape, sub_noise.shape)

            latents = slerp(subseed_strength, noise, sub_noise)

            logger.info("DEFORUM DIFFUSERS TXT2IMG")
            image = self.pipe(
                prompt=prompt,
                latent=latents,
                width=width,
                height=height,
                guidance_scale=scale,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                output_type="pil",
            ).images[0]
        else:
            logger.info("DEFORUM DIFFUSERS IMG2IMG")
            # self.img2img_pipe._guidance_scale = guidance_scale
            # self.img2img_pipe._guidance_rescale = guidance_rescale
            self.img2img_pipe._clip_skip = 2
            #self.img2img_pipe._cross_attention_kwargs = cross_attention_kwargs
            self.img2img_pipe._denoising_end = None
            self.img2img_pipe._denoising_start = None
            self.img2img_pipe._interrupt = False
            image = self.img2img_pipe.image_processor.preprocess(np.array(init_image).astype(np.uint8) / 255.0)
            latents, timesteps = self.prepare_latents_with_subseed(image, steps, strength, seed, subseed, subseed_strength)
            image = self.img2img_pipe(
                prompt=prompt,
                strength=strength,
                image=latents,
                width=width,
                height=height,
                guidance_scale=scale,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                output_type="pil",
                #timesteps=timesteps
            ).images[0]
        return image

    def cleanup(self):
        try:
            self.pipe.to('cpu')
            self.img2img_pipe.to('cpu')
            del self.pipe, self.img2img_pipe
        except:
            pass