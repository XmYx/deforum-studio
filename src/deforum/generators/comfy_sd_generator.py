import math
import os
import platform
import re
import secrets
import time

import numpy as np
import torch
from PIL import Image

from .comfy_utils import ensure_comfy
from .rng_noise_generator import ImageRNGNoise, slerp
from deforum.utils.deforum_cond_utils import blend_tensors
from deforum.utils.logging_config import logger
from ..utils.constants import config
from ..utils.model_download import (
    download_from_civitai,
    download_from_civitai_by_version_id,
)

cfg_guider = None


def simple_encode(self, pixel_samples):
    # Crop pixels to ensure dimensions are divisible by the downscale ratio
    x = (pixel_samples.shape[1] // self.downscale_ratio) * self.downscale_ratio
    y = (pixel_samples.shape[2] // self.downscale_ratio) * self.downscale_ratio
    if pixel_samples.shape[1] != x or pixel_samples.shape[2] != y:
        x_offset = (pixel_samples.shape[1] % self.downscale_ratio) // 2
        y_offset = (pixel_samples.shape[2] % self.downscale_ratio) // 2
        pixel_samples = pixel_samples[:, x_offset:x + x_offset, y_offset:y + y_offset, :]

    # Change channel order from (H, W, C) to (C, H, W)
    pixel_samples = pixel_samples.permute(0, 3, 1, 2).to(self.vae_dtype).to(self.device)
    pixel_samples = self.process_input(pixel_samples)
    latent_samples = self.first_stage_model.encode(pixel_samples)
    return latent_samples


def simple_decode(self, latent_samples):
    latent_samples = latent_samples.to(self.vae_dtype).to(self.device)
    decoded_samples = self.first_stage_model.decode(latent_samples)
    decoded_samples = self.process_output(decoded_samples)

    # Change channel order back from (C, H, W) to (H, W, C)
    decoded_samples = decoded_samples.permute(0, 2, 3, 1).to(self.output_device)
    return decoded_samples


def replace_encode_decode(vae):
    vae.encode = simple_encode.__get__(vae)
    vae.decode = simple_decode.__get__(vae)
class HIJackCFGGuider:
    def __init__(self, model_patcher):
        # print("BIG HOOOORAAAAY\n\n\n\n\n")
        self.model_patcher = model_patcher
        self.inner_model = self.model_patcher.model
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        import comfy

        for k in conds:
            self.original_conds[k] = comfy.sampler_helpers.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        import comfy

        return comfy.samplers.sampling_function(
            self.model_patcher.model,
            x,
            timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            self.cfg,
            model_options=model_options,
            seed=seed,
        )

    def inner_sample(
        self,
        noise,
        latent_image,
        device,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
    ):
        import comfy

        if (
            latent_image is not None and torch.count_nonzero(latent_image) > 0
        ):  # Don't shift the empty latent image.
            latent_image = self.model_patcher.model.process_latent_in(latent_image)

        self.conds = comfy.samplers.process_conds(
            self.model_patcher.model,
            noise,
            self.conds,
            device,
            latent_image,
            denoise_mask,
            seed,
        )

        extra_args = {"model_options": self.model_options, "seed": seed}

        samples = sampler.sample(
            self,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
        return self.model_patcher.model.process_latent_out(samples.to(torch.float16))

    def sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        self.conds = self.original_conds
        device = torch.device("cuda")
        sigmas = sigmas.to(device)
        output = self.inner_sample(
            noise, latent_image, device, sampler, sigmas, denoise_mask, None, True, seed
        )
        return output


def sampleDeforum(
    model,
    noise,
    positive,
    negative,
    cfg,
    device,
    sampler,
    sigmas,
    model_options={},
    latent_image=None,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    global cfg_guider
    if cfg_guider is None:
        cfg_guider = HIJackCFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(
        noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed
    )


class ComfyDeforumGenerator:
    def __init__(self, model_path: str = None, *args, **kwargs):
        ensure_comfy()
        import comfy.samplers
        import comfy
        comfy.samplers.CFGGuider = HIJackCFGGuider
        comfy.samplers.sample = sampleDeforum
        self.optimize = None
        import importlib
        # Construct the module name based on your package structure
        module_name = 'ComfyUI.custom_nodes.ComfyUI_smZNodes.__init__'
        # Import the module dynamically
        init_module = importlib.import_module(module_name)
        logger.info("Init module: " + str(init_module))
        self.clip_skip = 0
        self.device = "cuda"
        self.prompt = ""
        self.n_prompt = ""
        self.cond = None
        self.n_cond = None
        self.model = None
        self.clip = None
        self.vae = None
        self.pipe = None
        self.loaded_lora = None
        self.model_loaded = None
        self.model_path = model_path
        self.onediff_avail = False
        self.pipeline_type = "comfy"
        self.rng = None
        self.optimized = False
        self.torch_compile_vae = False
        self.load_taesdxl = False
    def optimize_model(self):
        from nodes import NODE_CLASS_MAPPINGS

        sfast_node = NODE_CLASS_MAPPINGS["ApplyStableFastUnet"]()
        self.model = sfast_node.apply_stable_fast(self.model, True)[0]
        self.optimized = True
        logger.info("Applied Stable-Fast Unet patch.")

    @torch.inference_mode()
    def encode_latent(
        self,
        vae,
        latent,
        seed,
        subseed,
        subseed_strength,
        seed_resize_from_h,
        seed_resize_from_w,
        reset_noise=False,
    ):

        ## TODO this looks wrong! Why override the supplied subseed strength?
        # subseed_strength = 0.6
        latent = latent.movedim(-1, 1)
        return {"samples": self.vae.first_stage_model.encode(latent.half().cuda() * 2.0 - 1.0)}

    def generate_latent(
        self,
        width,
        height,
        seed,
        subseed,
        subseed_strength,
        seed_resize_from_h=None,
        seed_resize_from_w=None,
        reset_noise=False,
    ):
        shape = [4, height // 8, width // 8]
        if self.rng is None or reset_noise:
            self.rng = ImageRNGNoise(shape=shape, seeds=[seed], subseeds=[subseed], subseed_strength=subseed_strength,
                                     seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
        noise = self.rng.first()
        # noise = torch.zeros([1, 4, height // 8, width // 8])
        return {"samples": noise.to("cuda")}

    def get_conds(self, clip, prompt, width, height, target_width, target_height):
        if not hasattr(self, "clip_node"):
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
        if self.vae is None:
            import comfy.sd
            from nodes import NODE_CLASS_MAPPINGS

            self.model, self.clip, self.vae, self.clipvision = (
                comfy.sd.load_checkpoint_guess_config(
                    self.model_path,
                    output_vae=True,
                    output_clip=True,
                    embedding_directory="models/embeddings",
                    output_clipvision=False,
                )
            )
            if self.load_taesdxl:
                vae_loader = NODE_CLASS_MAPPINGS['VAELoader']()
                self.cheap_vae = vae_loader.load_vae('taesdxl')[0]
                self.cheap_vae.first_stage_model.cuda()

            if 'linux' in platform.platform().lower() and self.torch_compile_vae:
                self.vae.first_stage_model.encode = torch.compile(self.vae.first_stage_model.encode, mode='reduce-overhead')
                self.vae.first_stage_model.decode = torch.compile(self.vae.first_stage_model.decode, mode='reduce-overhead')
            self.clip.patcher.offload_device = torch.device("cuda")
            self.vae.patcher.offload_device = torch.device("cuda")
            self.vae.first_stage_model.cuda()
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
            self.clip.clip_layer(-2)
            if config.enable_onediff:
                try:
                    from custom_nodes.onediff_comfy_nodes._nodes import BasicBoosterExecutor
                    from custom_nodes.onediff_comfy_nodes.modules import BoosterScheduler
                    # from custom_nodes.onediff_comfy_nodes.modules.oneflow.booster_quantization import \
                    #     OnelineQuantizationBoosterExecutor
                    custom_booster = BoosterScheduler(BasicBoosterExecutor())
                    logger.info("Starting to load onediff model...")
                    start_time = time.time()
                    self.model = custom_booster(self.model, ckpt_name=self.model_path)
                    logger.info(f"Onediff model loaded in: {time.time() - start_time}")
                    # self.vae = BoosterScheduler(BasicBoosterExecutor())(self.vae, ckpt_name=self.model_path)
                    self.model.weight_inplace_update = True
                    self.onediff_avail = True
                except:
                    logger.warn("NOT using onediff due to initialisation error. If you meant to, please check onediff custom nodes and their deps are correctly installed. To hid this message, set ENABLE_ONEDIFF=false.", exc_info=True)
            else:
                logger.info("NOT using onediff. If you meant to, set ENABLE_ONEDIFF=true", exc_info=True)
                

            self.model_loaded = True
            # from ..optimizations.deforum_comfy_trt.deforum_trt_comfyunet import TrtUnet
            # self.model.model.diffusion_model = TrtUnet()

    def load_lora_from_civitai(self, lora_id="", model_strength=0.0, clip_strength=0.0):

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
        except:
            print("LORA FAILED")

    def load_lora(self, model, clip, lora_path, strength_model, strength_clip):
        import comfy

        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

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
        *args,
        **kwargs,
    ):

        if not self.model_loaded:
            self.load_model()
            # self.load_lora_from_civitai('424720', 1.0, 1.0)
        if self.optimize and not (self.optimized or self.onediff_avail):
            try:
                self.optimize_model()
                self.optimized = True
            except:
                self.optimize = False
                self.optimized = False
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
                latent,
                seed,
                subseed,
                subseed_strength,
                seed_resize_from_h,
                seed_resize_from_w,
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
        # logger.info(f"seed/subseed/subseed_str={seed}/{subseed}/{subseed_strength}; strength={strength}; scale={scale}; sampler_name={sampler_name}; scheduler={scheduler};")
        if not hasattr(self, "sampler_node"):
            from nodes import NODE_CLASS_MAPPINGS
            self.sampler_node = NODE_CLASS_MAPPINGS['KSampler //Inspire']()
        steps = round(denoise * steps)
        if hasattr(self.sampler_node, "sample"):
            sample_fn = self.sampler_node.sample
        elif hasattr(self.sampler_node, "doit"):
            sample_fn = self.sampler_node.doit
        #logger.info(f"SEED:{seed}, STPS:{steps}, CFG:{scale}, SMPL:{sampler_name}, SCHD:{scheduler}, DENOISE:{denoise}, STR:{strength}, SUB:{subseed}, SUBSTR:{subseed_strength}")
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
        # sample = [{"samples": sample["samples"]}]
        # if sample[0]["samples"].shape[0] == 1:
        decoded = self.decode_sample(self.vae, sample["samples"])
        # Convert the decoded tensor to uint8 directly on the GPU
        np_array = torch.clamp(255.0 * decoded, 0, 255).byte().cpu().numpy()[0]
        # Convert the numpy array to a PIL image
        image = Image.fromarray(np_array)
        image = image.convert("RGB")
        if return_latent:
            return sample[0]["samples"], image
        else:
            return image
        # else:
        #     logger.info("decoding multi images")
        #     images = []
        #     x_samples = self.decode_sample(sample[0]["samples"])
        #     for sample in x_samples:
        #         np_array = np.clip(255.0 * sample.cpu().numpy(), 0, 255).astype(
        #             np.uint8
        #         )
        #         image = Image.fromarray(np_array)
        #         image = image.convert("RGB")
        #         images.append(image)
        #     return images
    @torch.inference_mode()
    def decode_sample(self, vae, sample):
        #     #sample = sample.to(torch.float16)
        #     decoded = vae.decode(sample).detach()
        sample = vae.first_stage_model.decode(sample)
        sample = torch.clamp((sample + 1.0) / 2.0, min=0.0, max=1.0)
        sample = sample.movedim(1, -1)
        return sample

    def cleanup(self):
        self.optimized = False
        return
        self.model.unpatch_model(device_to="cpu")
        self.vae.first_stage_model.to("cpu")
        # self.clip.to('cpu')
        del self.model
        del self.vae
        del self.clip


def common_ksampler_with_custom_noise(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
    noise=None,
):
    latent_image = latent["samples"]
    if noise is not None:
        noise = noise.next()  # .detach().cpu()
        # noise = rng_noise.clone()
    else:
        if disable_noise:
            noise = torch.zeros(
                latent_image.size(),
                dtype=latent_image.dtype,
                layout=latent_image.layout,
                device="cpu",
            )
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            from comfy.sample import prepare_noise

            noise = prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    # callback = latent_preview.prepare_callback(model, steps)
    # disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    from comfy.sample import sample as sample_k

    samples = sample_k(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=None,
        disable_pbar=False,
        seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples

    return (out,)


def apply_controlnet(conditioning, control_net, image, strength):
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


def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


NOISE_LEVELS = {
    "SD1": [
        14.6146412293,
        6.4745760956,
        3.8636745985,
        2.6946151520,
        1.8841921177,
        1.3943805092,
        0.9642583904,
        0.6523686016,
        0.3977456272,
        0.1515232662,
        0.0291671582,
    ],
    "SDXL": [
        14.6146412293,
        6.3184485287,
        3.7681790315,
        2.1811480769,
        1.3405244945,
        0.8620721141,
        0.5550693289,
        0.3798540708,
        0.2332364134,
        0.1114188177,
        0.0291671582,
    ],
    "SVD": [
        700.00,
        54.5,
        15.886,
        7.977,
        4.248,
        1.789,
        0.981,
        0.403,
        0.173,
        0.034,
        0.002,
    ],
}


def get_sigmas(model_type, steps):
    sigmas = NOISE_LEVELS[model_type][:]
    if (steps + 1) != len(sigmas):
        sigmas = loglinear_interp(sigmas, steps + 1)

    sigmas[-1] = 0
    return torch.FloatTensor(sigmas)


def sample_with_subseed(
    model,
    latent_image,
    main_seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    variation_strength,
    variation_seed,
    denoise,
    rng=None,
    sigmas=None,
):
    from nodes import common_ksampler as ksampler
    import comfy

    if main_seed == variation_seed:
        variation_seed += 1

    force_full_denoise = True
    disable_noise = True

    device = comfy.model_management.get_torch_device()

    # Generate base noise
    batch_size, _, height, width = latent_image["samples"].shape
    if rng is None:
        generator = torch.manual_seed(main_seed)
        base_noise = (
            torch.randn(
                (1, 4, height, width),
                dtype=torch.float16,
                device="cpu",
                generator=generator,
            )
            .repeat(batch_size, 1, 1, 1)
            .cpu()
        )

        # Generate variation noise
        generator = torch.manual_seed(variation_seed)
        variation_noise = torch.randn(
            (batch_size, 4, height, width),
            dtype=torch.float16,
            device="cpu",
            generator=generator,
        ).cpu()
        slerp_noise = slerp(variation_strength, base_noise, variation_noise)
        slerp_noise = slerp_noise.to("cuda")
    else:
        shape = [4, height, width]
        rng_noise = rng(
            shape=shape,
            seeds=[main_seed],
            subseeds=[variation_seed],
            subseed_strength=variation_strength,
            seed_resize_from_h=1024,
            seed_resize_from_w=1024,
        )
        slerp_noise = rng_noise.first()

    # Calculate sigma
    # comfy.model_management.load_model_gpu(model)
    sampler = comfy.samplers.KSampler(
        model,
        steps=steps,
        device=device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        model_options=model.model_options,
    )
    if sigmas is None:
        sigmas = sampler.sigmas
        end_at_step = steps  # min(steps, end_at_step)
        start_at_step = round(end_at_step - end_at_step * denoise)
        # sigmas = get_sigmas("SDXL", steps)
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
    else:
        sigma = sigmas[0]
        start_at_step = 0
        end_at_step = len(sigmas)
    sigma /= model.model.latent_format.scale_factor
    sigma = sigma.detach().cpu().item()
    work_latent = latent_image.copy()
    work_latent["samples"] = (
        latent_image["samples"].clone().to("cuda") + slerp_noise.to("cuda") * sigma
    )

    # # if there's a mask we need to expand it to avoid artifacts, 5 pixels should be enough
    # if "noise_mask" in latent_image:
    #     noise_mask = prepare_mask(latent_image["noise_mask"], latent_image['samples'].shape)
    #     work_latent["samples"] = noise_mask * work_latent["samples"] + (1-noise_mask) * latent_image["samples"]
    #     work_latent['noise_mask'] = expand_mask(latent_image["noise_mask"].clone(), 5, True)
    return ksampler(
        model,
        main_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        work_latent,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_at_step,
        last_step=end_at_step,
        force_full_denoise=force_full_denoise,
    )



def prepare_noise(latent_image, seed, noise_inds=None, noise_device="cpu", incremental_seed_mode="comfy", variation_seed=None, variation_strength=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    logger.info("This is surely our latent preparation and is way cooler \n\n\n\n")
    latent_size = latent_image.size()
    latent_size_1batch = [1, latent_size[1], latent_size[2], latent_size[3]]

    if variation_strength is not None and variation_strength > 0 or incremental_seed_mode.startswith("variation str inc"):
        if noise_device == "cpu":
            variation_generator = torch.manual_seed(variation_seed)
        else:
            torch.cuda.manual_seed(variation_seed)
            variation_generator = None

        variation_latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout,
                                       generator=variation_generator, device=noise_device)
    else:
        variation_latent = None

    def apply_variation(input_latent, strength_up=None):
        if variation_latent is None:
            return input_latent
        else:
            strength = variation_strength

            if strength_up is not None:
                strength += strength_up

            variation_noise = variation_latent.expand(input_latent.size()[0], -1, -1, -1)
            mixed_noise = (1 - strength) * input_latent + strength * variation_noise

            # NOTE: Since the variance of the Gaussian noise in mixed_noise has changed, it must be corrected through scaling.
            scale_factor = math.sqrt((1 - strength) ** 2 + strength ** 2)
            corrected_noise = mixed_noise / scale_factor

            return corrected_noise

    # method: incremental seed batch noise
    if noise_inds is None and incremental_seed_mode == "incremental":
        batch_cnt = latent_size[0]

        latents = None
        for i in range(batch_cnt):
            if noise_device == "cpu":
                generator = torch.manual_seed(seed+i)
            else:
                torch.cuda.manual_seed(seed+i)
                generator = None

            latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout,
                                 generator=generator, device=noise_device)

            latent = apply_variation(latent)

            if latents is None:
                latents = latent
            else:
                latents = torch.cat((latents, latent), dim=0)

        return latents

    # method: incremental variation batch noise
    elif noise_inds is None and incremental_seed_mode.startswith("variation str inc"):
        batch_cnt = latent_size[0]

        latents = None
        for i in range(batch_cnt):
            if noise_device == "cpu":
                generator = torch.manual_seed(seed)
            else:
                torch.cuda.manual_seed(seed)
                generator = None

            latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout,
                                 generator=generator, device=noise_device)

            step = float(incremental_seed_mode[18:])
            latent = apply_variation(latent, step*i)

            if latents is None:
                latents = latent
            else:
                latents = torch.cat((latents, latent), dim=0)

        return latents

    # method: comfy batch noise
    if noise_device == "cpu":
        generator = torch.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
        generator = None

    if noise_inds is None:
        latents = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                              generator=generator, device=noise_device)
        latents = apply_variation(latents)
        return latents

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout,
                            generator=generator, device=noise_device)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises
