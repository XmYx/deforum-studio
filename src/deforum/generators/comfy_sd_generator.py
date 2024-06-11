"""
comfy_deforum_generator.py

This module provides the ComfyDeforumGenerator class and related functions for generating images using
a pre-trained model. The class supports various optimizations, latent encoding, and integration with IP adapter
models and LORA models from CivitAI. The generator can be used to produce high-quality images based on given
prompts and other conditioning factors.
"""
import os
import secrets
import numpy as np
import torch
from PIL import Image

from .comfy_utils import ensure_comfy
from .rng_noise_generator import ImageRNGNoise
from deforum.utils.deforum_cond_utils import blend_tensors
from deforum.utils.logging_config import logger
from ..utils.constants import config
from ..utils.model_download import (
    download_from_civitai_by_version_id,
)

def pil2tensor(image):
    """
    Converts a PIL Image to a PyTorch tensor.

    Args:
        image (PIL.Image): The image to convert.

    Returns:
        torch.Tensor: The converted image tensor.
    """
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class ComfyDeforumGenerator:
    """
    A class to generate images using a pre-trained model with various optimizations and conditioning options.

    Attributes:
        model_path (str): The path to the pre-trained model.
        model (torch.nn.Module): The loaded model.
        clip (torch.nn.Module): The CLIP model.
        vae (torch.nn.Module): The VAE model.
        ip_adapter (torch.nn.Module): The IP adapter model.
        optimizations (list): List of available optimizations.
        rng (ImageRNGNoise): Random noise generator for image creation.
    """
    def __init__(self, model_path: str = None, *args, **kwargs):
        """
        Initializes the ComfyDeforumGenerator with optional model path and other parameters.

        Args:
            model_path (str, optional): The path to the pre-trained model. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.model_path = model_path
        ensure_comfy()
        self.cond = None
        self.n_cond = None
        self.model = None
        self.clip = None
        self.vae = None
        self.ip_loader_node = None
        self.clip_vision_loader_node = None
        self.ip_adapter_apply_node = None
        self.ip_adapter = None
        self.clip_vision = None
        self.clip_node = None
        self.sampler_node = None
        self.pipe = None
        self.loaded_lora = None
        self.model_loaded = None
        self.rng = None
        self.optimized = False
        self.current_optimization = ""
        self.prompt = ""
        self.n_prompt = ""
        self.device = "cuda"
        self.optimizations = []
        self.initialize_optimizations()

    def initialize_optimizations(self):
        """
        Initializes available optimizations for the model.
        """
        from nodes import NODE_CLASS_MAPPINGS
        try:
            sfast_node = NODE_CLASS_MAPPINGS["ApplyStableFastUnet"]()
            self.optimizations.append('stable_fast')
            logger.info("Stable Fast optimization available")
            del sfast_node

        except Exception as e:
            logger.info("Stable Fast optimization not available")
            logger.debug(str(e))

        try:
            from custom_nodes.onediff_comfy_nodes._nodes import BasicBoosterExecutor
            self.optimizations.append('onediff')
            logger.info("Onediff optimization available")

        except Exception as e:
            logger.info("Onediff optimization not available")
            logger.debug(str(e))

    def optimize_model(self, optimization: str = 'stable_fast'):
        """
        Applies the specified optimization to the model.

        Args:
            optimization (str): The name of the optimization to apply. Defaults to 'stable_fast', supports 'onediff' too.
        """
        if self.optimized and optimization in self.optimizations and self.current_optimization != optimization:
            self.cleanup()
        if self.current_optimization != optimization and optimization in self.optimizations:
            opt_method = getattr(self, f'apply_{optimization}')
            self.model = opt_method(self.model, self.model_path)
            self.optimized = True
            self.current_optimization = optimization

    @classmethod
    def apply_stable_fast(cls, model, *args, **kwargs):
        """
        Applies the Stable-Fast Unet patch to the model.

        Args:
            model (torch.nn.Module): The model to optimize.

        Returns:
            torch.nn.Module: The optimized model.
        """
        from nodes import NODE_CLASS_MAPPINGS
        sfast_node = NODE_CLASS_MAPPINGS["ApplyStableFastUnet"]()
        model = sfast_node.apply_stable_fast(model, True)[0]
        logger.info("Applied Stable-Fast Unet patch.")
        return model

    @classmethod
    def apply_onediff(cls, model, model_path, *args, **kwargs):
        """
        Applies the Onediff patch to the model.

        Args:
            model (torch.nn.Module): The model to optimize.

        Returns:
            torch.nn.Module: The optimized model.
        """
        from custom_nodes.onediff_comfy_nodes._nodes import BasicBoosterExecutor
        from custom_nodes.onediff_comfy_nodes.modules import BoosterScheduler
        custom_booster = BoosterScheduler(BasicBoosterExecutor())
        model = custom_booster(model, ckpt_name=model_path)
        model.weight_inplace_update = True
        return model

    @torch.inference_mode()
    def set_ip_adapter_image(self,
                             image:Image.Image,
                             weight:float=1.0,
                             start:float=0.0,
                             end:float=1.0):
        """
        Set the IP adapter image with specified parameters.

        Args:
            image: The image to set.
            weight (float): The weight for the IP adapter. Defaults to 1.0.
            start (float): The start position. Defaults to 0.0.
            end (float): The end position. Defaults to 1.0.
        """
        if self.model_loaded and self.optimized:
            self.cleanup()
        if not self.model_loaded:
            self.load_model()
        from nodes import NODE_CLASS_MAPPINGS
        if self.ip_loader_node is None:
            self.ip_loader_node = NODE_CLASS_MAPPINGS['IPAdapterModelLoader']()
            self.clip_vision_loader_node = NODE_CLASS_MAPPINGS['CLIPVisionLoader']()
            self.ip_adapter_apply_node = NODE_CLASS_MAPPINGS['IPAdapterAdvanced']()
            self.ip_adapter = self.ip_loader_node.load_ipadapter_model('ip-adapter-plus_sdxl_vit-h.safetensors')[0]
            self.clip_vision = self.clip_vision_loader_node.load_clip('CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors')[0]

        img_tensor = pil2tensor(image)

        self.model = self.ip_adapter_apply_node.apply_ipadapter(self.model,
                                                                self.ip_adapter,
                                                                clip_vision=self.clip_vision,
                                                                image=img_tensor,
                                                                weight=weight,
                                                                weight_type='linear',
                                                                combine_embeds='concat',
                                                                start_at=start,
                                                                end_at=end,
                                                                embeds_scaling='V only')[0]

    @torch.inference_mode()
    def encode_latent(
            self,
            vae,
            latent,
    ):
        """
        Encode a latent tensor using the VAE model.

        Args:
            vae: The VAE model.
            latent: The latent tensor to encode.

        Returns:
            dict: Encoded samples.
        """
        latent = latent.movedim(-1, 1)
        return {"samples": vae.first_stage_model.encode(latent.to(torch.bfloat16).cuda() * 2.0 - 1.0)}

    def generate_latent(
            self,
            width: int = 1024,
            height: int = 1024,
            seed: int = 0,
            subseed: int = 0,
            subseed_strength: float = 0.1,
            seed_resize_from_h=None,
            seed_resize_from_w=None,
            reset_noise=False,
    ):
        """
        Generate a latent tensor with specified dimensions and seed.

        Args:
            width (int): The width of the latent tensor. Defaults to 1024.
            height (int): The height of the latent tensor. Defaults to 1024.
            seed (int): The seed for the random number generator. Defaults to 0.
            subseed (int): The subseed for additional randomness. Defaults to 0.
            subseed_strength (float): The strength of the subseed. Defaults to 0.1.
            seed_resize_from_h: Resize height from seed. Defaults to None.
            seed_resize_from_w: Resize width from seed. Defaults to None.
            reset_noise (bool): Flag to reset noise. Defaults to False.

        Returns:
            dict: Generated samples.
        """
        shape = [4, height // 8, width // 8]
        if self.rng is None or reset_noise:
            self.rng = ImageRNGNoise(shape=shape, seeds=[seed], subseeds=[subseed], subseed_strength=subseed_strength,
                                     seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
        noise = self.rng.first()
        return {"samples": noise.to("cuda")}

    def get_conds(self,
                  clip,
                  prompt,
                  width,
                  height,
                  target_width,
                  target_height):
        """
        Get conditioning tensors from the CLIP model based on the prompt and dimensions.

        Args:
            clip: The CLIP model.
            prompt (str): The text prompt for conditioning.
            width (int): The width for conditioning.
            height (int): The height for conditioning.
            target_width (int): The target width.
            target_height (int): The target height.

        Returns:
            Conditioning tensors.
        """
        if self.clip_node is None:
            from nodes import NODE_CLASS_MAPPINGS
            self.clip_node = NODE_CLASS_MAPPINGS["smZ CLIPTextEncode"]()
        conds = self.clip_node.encode(
            clip,
            prompt,
            parser="A1111",
            mean_normalization=True,
            multi_conditioning=True,
            use_old_emphasis_implementation=False,
            with_SDXL=False,
            ascore=6.0,
            width=width,
            height=height,
            crop_w=0,
            crop_h=0,
            target_width=target_width,
            target_height=target_height,
            text_g=prompt,
            text_l=prompt,
            smZ_steps=1,
        )[0]
        return conds

    def load_model(self):
        """
        Load the model, CLIP, and VAE components.
        """
        if self.vae is None or not self.model_loaded:
            import comfy.sd
            from nodes import NODE_CLASS_MAPPINGS
            self.model, self.clip, self.vae, _ = (
                comfy.sd.load_checkpoint_guess_config(
                    self.model_path,
                    output_vae=True,
                    output_clip=True,
                    embedding_directory="models/embeddings",
                    output_clipvision=False,
                )
            )
            self.clip.patcher.offload_device = torch.device("cuda")
            self.vae.patcher.offload_device = torch.device("cuda")
            self.vae.first_stage_model.cuda()
            self.clip.clip_layer(-2)
            self.apply_smz_optimizations()

    def apply_smz_optimizations(self):
        """
        Apply SMZ optimizations to the model and CLIP.
        """
        from nodes import NODE_CLASS_MAPPINGS

        settings_node = NODE_CLASS_MAPPINGS["smZ Settings"]()
        settings_dict = {}
        for k, v in settings_node.INPUT_TYPES()["optional"].items():
            if "default" in v[1]:
                settings_dict[k] = v[1]["default"]
        settings_dict["RNG"] = "gpu"
        settings_dict["pad_cond_uncond"] = True
        settings_dict["Use CFGDenoiser"] = True
        settings_dict["disable_nan_check"] = True
        settings_dict["upcast_sampling"] = False
        settings_dict["batch_cond_uncond"] = True
        settings_dict["sgm_noise_multiplier"] = False
        settings_dict["enable_emphasis"] = True
        settings_dict["ENSD"] = 0
        settings_dict["s_noise"] = 1.0
        settings_dict["eta"] = 1.0
        settings_dict["s_churn"] = 0.0
        settings_dict["t_min"] = 0.0
        settings_dict["t_max"] = 0.0
        self.model = settings_node.run(self.model, **settings_dict)[0]
        self.clip = settings_node.run(self.clip, **settings_dict)[0]
        self.model_loaded = True

    def load_lora_from_civitai(self,
                               lora_id="",
                               model_strength=0.0,
                               clip_strength=0.0):
        """
        Load a LoRA model from CivitAI.

        Args:
            lora_id (str): The ID of the LoRA model.
            model_strength (float): Strength of the model. Defaults to 0.0.
            clip_strength (float): Strength of the CLIP model. Defaults to 0.0.
        """
        cache_dir = os.path.join(config.root_path, "models")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            import comfy
            filename = download_from_civitai_by_version_id(
                model_id=lora_id, destination=cache_dir, force_download=False
            )
            lora_path = os.path.join(cache_dir, filename)
            lora = None
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == lora_path:
                    lora = self.loaded_lora[1]
                else:
                    temp = self.loaded_lora
                    self.loaded_lora = None
                    del temp

            if lora is None:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)

            self.model, self.clip = comfy.sd.load_lora_for_models(
                self.model, self.clip, lora, model_strength, clip_strength
            )
        except Exception as e:
            logger.info(
                f'LORA Loading failed from CivitAI, please check wether the version id: {str(lora_id)} is correct')
            logger.debug(str(e))

    def load_lora(self,
                  model,
                  clip,
                  lora_path,
                  strength_model,
                  strength_clip):
        """
        Load a LoRA model from a specified path.

        Args:
            model: The model to apply the LoRA to.
            clip: The CLIP model.
            lora_path (str): The path to the LoRA file.
            strength_model (float): The strength to apply to the model.
            strength_clip (float): The strength to apply to the CLIP model.

        Returns:
            Tuple of the model and CLIP with the applied LoRA.
        """
        import comfy
        if strength_model == 0 and strength_clip == 0:
            return model, clip
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return model_lora, clip_lora

    @torch.inference_mode()
    def __call__(
            self,
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
            subseed_strength=0.0,
            cnet_image=None,
            cond=None,
            n_cond=None,
            return_latent=None,
            latent=None,
            last_step=None,
            seed_resize_from_h=0,
            seed_resize_from_w=0,
            reset_noise=False,
            enable_prompt_blend=True,
            use_areas=False,
            areas=None,
            use_optimization=None,
            *args,
            **kwargs,
    ):
        """
        Generate an image or latent based on the provided parameters.

        Args:
            prompt (str): The main text prompt.
            pooled_prompts: Precomputed conditioning tensors.
            next_prompt (str): The next prompt for blending.
            prompt_blend (float): Blending factor for prompts.
            negative_prompt (str): The negative prompt for conditioning.
            steps (int): Number of sampling steps. Defaults to 25.
            scale (float): Scale for the CFG. Defaults to 7.5.
            sampler_name (str): Name of the sampler. Defaults to 'dpmpp_2m_sde'.
            scheduler (str): Name of the scheduler. Defaults to 'karras'.
            width: Width of the generated image.
            height: Height of the generated image.
            seed (int): Seed for random generation. Defaults to -1.
            strength (float): Strength of the initial image. Defaults to 1.0.
            init_image: Initial image for conditioning.
            subseed (int): Subseed for additional randomness. Defaults to -1.
            subseed_strength (float): Strength of the subseed. Defaults to 0.0.
            cnet_image: ControlNet image for additional conditioning.
            cond: Conditioning tensors.
            n_cond: Negative conditioning tensors.
            return_latent: Flag to return latent instead of image.
            latent: Precomputed latent tensor.
            last_step: Last step of the process.
            seed_resize_from_h (int): Seed resize height. Defaults to 0.
            seed_resize_from_w (int): Seed resize width. Defaults to 0.
            reset_noise (bool): Flag to reset noise. Defaults to False.
            enable_prompt_blend (bool): Flag to enable prompt blending. Defaults to True.
            use_areas (bool): Flag to use specific areas for conditioning. Defaults to False.
            areas: Specific areas for conditioning.
            use_optimization: Optimization method to use.

        Returns:
            PIL.Image or tuple: Generated image or latent and image.
        """

        if not self.model_loaded:
            self.load_model()
            # self.load_lora_from_civitai('424720', 1.0, 1.0)

        if use_optimization:
            self.optimize_model(use_optimization)

        if seed_resize_from_h == 0:
            seed_resize_from_h = 1024
        if seed_resize_from_w == 0:
            seed_resize_from_w = 1024
        if seed == -1:
            seed = secrets.randbelow(18446744073709551615)

        denoise = 1 - strength if 0 < strength < 1 else 1.0
        subseed_strength = 0 if denoise == 1.0 else subseed_strength
        if subseed == -1:
            subseed = secrets.randbelow(18446744073709551615)
        if cnet_image is not None:
            cnet_image = torch.from_numpy(
                np.array(cnet_image).astype(np.float16) / 255.0
            ).unsqueeze(0)
        if init_image is None or reset_noise:
            logger.info(
                f"reset_noise: {reset_noise}; resetting denoise strength to 1.0 from: {denoise}"
            )
            denoise = 1.0
            if latent is None:
                if width is None:
                    width = 1024
                if height is None:
                    height = 960
                latent = self.generate_latent(
                    width,
                    height,
                    seed,
                    subseed,
                    subseed_strength,
                    seed_resize_from_h,
                    seed_resize_from_w,
                    reset_noise,
                )
            else:
                if isinstance(latent, torch.Tensor):
                    latent = {"samples": latent}
                elif isinstance(latent, list):
                    latent = {"samples": torch.stack(latent, dim=0)}
                else:
                    latent = latent
        else:
            latent = (
                    torch.from_numpy(np.array(init_image))
                    .cuda()
                    .unsqueeze(0)
                    .to(dtype=torch.float16)
                    / 255.0
            )
            latent = self.encode_latent(
                self.vae,
                latent
            )
        if self.prompt != prompt or self.cond is None:
            self.prompt = prompt
            if pooled_prompts is None and prompt is not None:
                self.cond = self.get_conds(
                    self.clip,
                    prompt,
                    width,
                    height,
                    seed_resize_from_w,
                    seed_resize_from_w,
                )
            elif pooled_prompts is not None:
                self.cond = pooled_prompts
        cond = self.cond
        if use_areas and areas is not None:
            from nodes import ConditioningSetArea
            area_setter = ConditioningSetArea()
            for area in areas:
                logger.info(f"AREA TO USE: {area}")
                prompt = area.get("prompt", None)
                if prompt:
                    new_cond = self.get_conds(
                        self.clip,
                        area["prompt"],
                        width,
                        height,
                        seed_resize_from_w,
                        seed_resize_from_w,
                    )
                    new_cond = area_setter.append(
                        conditioning=new_cond,
                        width=int(area["w"]),
                        height=int(area["h"]),
                        x=int(area["x"]),
                        y=int(area["y"]),
                        strength=area["s"],
                    )[0]
                    cond += new_cond
        if self.n_prompt != negative_prompt or self.n_cond is None:
            self.n_cond = self.get_conds(
                self.clip,
                negative_prompt,
                width,
                height,
                seed_resize_from_w,
                seed_resize_from_w,
            )
            self.n_prompt = negative_prompt
        if next_prompt is not None and enable_prompt_blend:
            if next_prompt != prompt and next_prompt != "":
                if 0.0 < prompt_blend < 1.0:
                    logger.info(f"[DEFORUM PROMPT BLEND] {prompt_blend},\n\n{prompt}\n{next_prompt}")

                    next_cond = self.get_conds(
                        self.clip,
                        next_prompt,
                        width,
                        height,
                        seed_resize_from_w,
                        seed_resize_from_w,
                    )
                    cond = blend_tensors(
                        cond[0], next_cond[0], blend_value=prompt_blend
                    )
        if cnet_image is not None:
            cond = apply_controlnet(cond, self.controlnet, cnet_image, 1.0)
        if self.sampler_node is None:
            from nodes import NODE_CLASS_MAPPINGS
            self.sampler_node = NODE_CLASS_MAPPINGS['KSampler //Inspire']()
        steps = round(denoise * steps)
        sample_fn = None
        if hasattr(self.sampler_node, "sample"):
            sample_fn = self.sampler_node.sample
        elif hasattr(self.sampler_node, "doit"):
            sample_fn = self.sampler_node.doit
        logger.debug(
            f"SEED:{seed}, STPS:{steps}, CFG:{scale},"
            f" SMPL:{sampler_name}, SCHD:{scheduler},"
            f" DENOISE:{denoise}, STR:{strength}, SUB:{subseed},"
            f" SUBSTR:{subseed_strength}")
        if sample_fn:
            sample = sample_fn(
                self.model,
                seed,
                steps,
                scale,
                sampler_name,
                scheduler,
                cond,
                self.n_cond,
                latent,
                denoise,
                noise_mode="GPU(=A1111)",
                batch_seed_mode="comfy",
                variation_seed=subseed,
                variation_strength=subseed_strength,
            )[0]
            decoded = self.decode_sample(self.vae, sample["samples"].to(torch.bfloat16).to('cuda'))
            # Convert the decoded tensor to uint8 directly on the GPU
            np_array = torch.clamp(255.0 * decoded, 0, 255).byte().cpu().numpy()[0]
            # Convert the numpy array to a PIL image
            image = Image.fromarray(np_array)
            image = image.convert("RGB")
            if return_latent:
                return sample[0]["samples"], image
            else:
                return image
        else:
            logger.debug("Error, no sampler function found")
            return None

    @torch.inference_mode()
    def decode_sample(self, vae, sample):
        """
        Decode a sample tensor using the VAE.

        Args:
            vae: The VAE model.
            sample: The sample tensor to decode.

        Returns:
            torch.Tensor: Decoded sample.
        """
        sample = vae.first_stage_model.decode(sample)
        sample = torch.clamp((sample + 1.0) / 2.0, min=0.0, max=1.0)
        sample = sample.movedim(1, -1)
        return sample

    def cleanup(self):
        """
        Cleanup resources and unload the model from the GPU.
        """
        self.optimized = False
        self.model.unpatch_model(device_to="cpu")
        self.vae.first_stage_model.to("cpu")
        del self.model
        del self.vae
        del self.clip
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def apply_controlnet(conditioning,
                     control_net,
                     image,
                     strength):
    """
    Apply ControlNet conditioning to the provided image.

    Args:
        conditioning: Existing conditioning tensors.
        control_net: The ControlNet model.
        image: The image for conditioning.
        strength (float): The strength of the ControlNet conditioning.

    Returns:
        Conditioning tensors with applied ControlNet.
    """
    with torch.inference_mode():
        if strength == 0:
            return (conditioning,)

        c = []
        control_hint = image.movedim(-1, 1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if "control" in t[1]:
                c_net.set_previous_controlnet(t[1]["control"])
            n[1]["control"] = c_net
            n[1]["control_apply_to_uncond"] = True
            c.append(n)
    return c
