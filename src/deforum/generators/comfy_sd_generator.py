import os
import re
import secrets

import numpy as np
import torch
from PIL import Image

from .comfy_utils import ensure_comfy
from .rng_noise_generator import ImageRNGNoise, slerp
from deforum.utils.deforum_cond_utils import blend_tensors
from deforum.utils.logging_config import logger
from ..utils.constants import root_path
from ..utils.model_download import (
    download_from_civitai,
    download_from_civitai_by_version_id,
)

cfg_guider = None


def prepare_sampling(model, noise_shape, conds):
    # device = model.load_device
    # real_model = None
    # models, inference_memory = get_additional_models(conds, model.model_dtype())
    # comfy.model_management.load_models_gpu([model] + models, model.memory_required(
    #     [noise_shape[0] * 2] + list(noise_shape[1:])) + inference_memory)
    # real_model = model.model

    return model.model, conds, []


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
        return self.model_patcher.model.process_latent_out(samples.to(torch.float32))

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
        # if sigmas.shape[-1] == 0:
        #     return latent_image
        #
        self.conds = self.original_conds
        import comfy

        # self.conds = {}
        # for k in self.original_conds:
        #     self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
        #
        # self.model_patcher.model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(
        #     self.model_patcher, noise.shape, self.conds)
        # device = self.model_patcher.load_device
        #
        # # if denoise_mask is not None:
        # #     denoise_mask = comfy.sampler_helpers.prepare_mask(denoise_mask, noise.shape, device)
        device = torch.device("cuda")
        # noise = noise.to(device)
        # latent_image = latent_image.to(device)
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
    # print("Hooray here too")
    if cfg_guider is None:
        cfg_guider = HIJackCFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(
        noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed
    )


class ComfyDeforumGenerator:

    def __init__(self, model_path: str = None, lcm=False, trt=False):

        ensure_comfy()
        import comfy.samplers
        import comfy

        comfy.samplers.CFGGuider = HIJackCFGGuider
        comfy.samplers.sample = sampleDeforum
        self.optimize = None
        # from deforum.datafunctions.ensure_comfy import ensure_comfy
        # ensure_comfy()
        # from deforum.datafunctions import comfy_functions
        # from comfy import model_management, controlnet

        # model_management.vram_state = model_management.vram_state.HIGH_VRAM
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
        # if not lcm:
        #     # if model_path is None:
        #     #     models_dir = os.path.join(default_cache_folder)
        #     #     fetch_and_download_model("125703", default_cache_folder)
        #     #     model_path = os.path.join(models_dir, "protovisionXLHighFidelity3D_release0620Bakedvae.safetensors")
        #     #     # model_path = os.path.join(models_dir, "SSD-1B.safetensors")
        #
        #     self.load_model(model_path, trt)
        #
        #     self.pipeline_type = "comfy"
        # if lcm:
        #     self.load_lcm()
        #     self.pipeline_type = "diffusers_lcm"

        # self.controlnet = controlnet.load_controlnet(model_name)
        self.pipeline_type = "comfy"
        self.rng = None
        self.optimized = False

    def optimize_model(self):
        from nodes import NODE_CLASS_MAPPINGS

        sfast_node = NODE_CLASS_MAPPINGS["ApplyStableFastUnet"]()
        self.model = sfast_node.apply_stable_fast(self.model, True)[0]
        self.optimized = True
        logger.info("Applied Stable-Fast Unet patch.")

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

        with torch.inference_mode():
            latent = latent.to(torch.float32)
            latent = vae.encode_tiled(latent[:, :, :, :3])
            latent = latent.to("cuda")
        # if self.rng is None or reset_noise:
        #     self.rng = ImageRNGNoise(shape=latent[0].shape, seeds=[seed], subseeds=[subseed], subseed_strength=subseed_strength,
        #                              seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
        #     noise = self.rng.first()
        # #     noise = slerp(subseed_strength, noise, latent)
        # else:
        #     noise = self.rng.next()
        #     noise = slerp(subseed_strength, noise, latent)
        #     noise = latent
        return {"samples": latent}

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
        # shape = [4, height // 8, width // 8]
        # if self.rng is None or reset_noise:
        #     self.rng = ImageRNGNoise(shape=shape, seeds=[seed], subseeds=[subseed], subseed_strength=subseed_strength,
        #                              seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
        # noise = self.rng.next()
        noise = torch.zeros([1, 4, height // 8, width // 8])
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
            multi_conditioning=False,
            use_old_emphasis_implementation=False,
            with_SDXL=True,
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
        # with torch.inference_mode():
        #     # clip.clip_layer(0)
        #     tokens = clip.tokenize(prompt)
        #     cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        #     return [[cond, {"pooled_output": pooled}]]

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
            #print(self.model.offload_device)
            self.clip.patcher.offload_device = torch.device("cuda")
            self.vae.patcher.offload_device = torch.device("cuda")
            #print(self.clip.patcher.offload_device)
            #print(self.vae.patcher.offload_device)
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
            self.model_loaded = True

            # from ..optimizations.deforum_comfy_trt.deforum_trt_comfyunet import TrtUnet
            # self.model.model.diffusion_model = TrtUnet()

    def load_lora_from_civitai(self, lora_id="", model_strength=0.0, clip_strength=0.0):

        cache_dir = os.path.join(root_path, "models")
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

    @staticmethod
    def load_lcm():
        logger.info("Deprecated for now")
        # from deforum.lcm.lcm_pipeline import LatentConsistencyModelPipeline
        #
        # from deforum.lcm.lcm_scheduler import LCMScheduler
        # self.scheduler = LCMScheduler.from_pretrained(
        #     os.path.join(root_path, "configs/lcm_scheduler.json"))
        #
        # self.pipe = LatentConsistencyModelPipeline.from_pretrained(
        #     pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
        #     scheduler=self.scheduler
        # ).to("cuda")
        # from deforum.lcm.lcm_i2i_pipeline import LatentConsistencyModelImg2ImgPipeline
        # # self.img2img_pipe = LatentConsistencyModelImg2ImgPipeline(
        # #     unet=self.pipe.unet,
        # #     vae=self.pipe.vae,
        # #     text_encoder=self.pipe.text_encoder,
        # #     tokenizer=self.pipe.tokenizer,
        # #     scheduler=self.pipe.scheduler,
        # #     feature_extractor=self.pipe.feature_extractor,
        # #     safety_checker=None,
        # # )
        # self.img2img_pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
        #     pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
        #     safety_checker=None,
        # ).to("cuda")

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
            # self.load_lora_from_civitai('413566', 1.0, 1.0)
        if seed_resize_from_h == 0:
            seed_resize_from_h = 1024
        if seed_resize_from_w == 0:
            seed_resize_from_w = 1024
        if seed == -1:
            seed = secrets.randbelow(18446744073709551615)
        if self.optimize:
            self.optimize_model()
            self.optimize = False


        if strength <= 0.05 or strength >= 1.0:
            strength = 1.0
            # reset_noise = True
            # init_image = None
        else:
            strength = 1 - strength if strength != 1.0 else strength

        if subseed == -1:
            subseed = secrets.randbelow(18446744073709551615)

        if cnet_image is not None:
            cnet_image = torch.from_numpy(
                np.array(cnet_image).astype(np.float32) / 255.0
            ).unsqueeze(0)

        if init_image is None or reset_noise:
            logger.info(
                f"reset_noise: {reset_noise}; resetting strength to 1.0 from: {strength}"
            )
            strength = 1.0
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

        # cond = []
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

        steps = round(strength * steps)
        # if subseed_strength > 0:
        #     subseed_strength = subseed_strength * 10

        if hasattr(self.sampler_node, "sample"):
            sample_fn = self.sampler_node.sample
        elif hasattr(self.sampler_node, "doit"):
            sample_fn = self.sampler_node.doit
        logger.info(f"SEED:{seed}, STPS:{steps}, CFG:{scale}, SMPL:{sampler_name}, SCHD:{scheduler}, STR:{strength}, SUB:{subseed}, SUBSTR:{subseed_strength}")
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
            strength,
            noise_mode="GPU(=A1111)",
            batch_seed_mode="comfy",
            variation_seed=subseed,
            variation_strength=subseed_strength,
        )[0]
        sample = [{"samples": sample["samples"]}]

        if sample[0]["samples"].shape[0] == 1:
            decoded = self.decode_sample(self.vae, sample[0]["samples"])
            np_array = np.clip(255.0 * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[
                0
            ]
            image = Image.fromarray(np_array)
            # image = Image.fromarray(np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0])
            image = image.convert("RGB")

            if return_latent:
                return sample[0]["samples"], image
            else:
                return image
        else:
            logger.info("decoding multi images")
            images = []
            x_samples = self.vae.decode_tiled(sample[0]["samples"])
            for sample in x_samples:
                np_array = np.clip(255.0 * sample.cpu().numpy(), 0, 255).astype(
                    np.uint8
                )
                image = Image.fromarray(np_array)
                # image = Image.fromarray(np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0])
                image = image.convert("RGB")
                images.append(image)

            return images

    def decode_sample(self, vae, sample):
        with torch.inference_mode():
            sample = sample.to(torch.float32)
            decoded = vae.decode(sample).detach()

        return decoded

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
                dtype=torch.float32,
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
            dtype=torch.float32,
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