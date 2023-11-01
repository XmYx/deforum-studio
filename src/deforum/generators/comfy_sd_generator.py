import secrets

import numpy as np
import torch
from PIL import Image

from .comfy_utils import ensure_comfy
from .rng_noise_generator import ImageRNGNoise
from ..utils.deforum_cond_utils import blend_tensors


class ComfyDeforumGenerator:

    def __init__(self, model_path: str = None, lcm=False, trt=False):

        ensure_comfy()

        # from deforum.datafunctions.ensure_comfy import ensure_comfy
        # ensure_comfy()
        # from deforum.datafunctions import comfy_functions
        # from comfy import model_management, controlnet

        # model_management.vram_state = model_management.vram_state.HIGH_VRAM
        self.clip_skip = -2
        self.device = "cuda"

        self.prompt = ""
        self.n_prompt = ""

        cond = None
        self.n_cond = None

        self.model = None
        self.clip = None
        self.vae = None
        self.pipe = None

        if not lcm:
            # if model_path is None:
            #     models_dir = os.path.join(default_cache_folder)
            #     fetch_and_download_model("125703", default_cache_folder)
            #     model_path = os.path.join(models_dir, "protovisionXLHighFidelity3D_release0620Bakedvae.safetensors")
            #     # model_path = os.path.join(models_dir, "SSD-1B.safetensors")

            self.load_model(model_path, trt)

            self.pipeline_type = "comfy"
        if lcm:
            self.load_lcm()
            self.pipeline_type = "diffusers_lcm"

        # self.controlnet = controlnet.load_controlnet(model_name)

        self.rng = None

    def encode_latent(self, latent):
        with torch.inference_mode():
            latent = latent.to(torch.float32)
            latent = self.vae.encode_tiled(latent[:, :, :, :3])
            latent = latent.to("cuda")

        return {"samples": latent}

    def generate_latent(self, width, height, seed, subseed, subseed_strength, seed_resize_from_h=None,
                        seed_resize_from_w=None, reset_noise=False):
        shape = [4, height // 8, width // 8]
        if self.rng is None or reset_noise:
            self.rng = ImageRNGNoise(shape=shape, seeds=[seed], subseeds=[subseed], subseed_strength=subseed_strength,
                                     seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
        noise = self.rng.next()
        # noise = torch.zeros([1, 4, width // 8, height // 8])
        return {"samples": noise}

    def get_conds(self, prompt):
        with torch.inference_mode():
            clip_skip = -2
            if self.clip_skip != clip_skip or self.clip.layer_idx != clip_skip:
                self.clip.layer_idx = clip_skip
                self.clip.clip_layer(clip_skip)
                self.clip_skip = clip_skip

            tokens = self.clip.tokenize(prompt)
            cond, pooled = self.clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]

    def load_model(self, model_path: str, trt: bool = False):

        import comfy.sd
        self.model, self.clip, self.vae, clipvision = (
            comfy.sd.load_checkpoint_guess_config(model_path,
                                                  output_vae=True,
                                                  output_clip=True,
                                                  embedding_directory="models/embeddings",
                                                  output_clipvision=False,
                                                  )
        )

        if trt:
            from ..optimizations.deforum_comfy_trt.deforum_trt_comfyunet import TrtUnet
            self.model.model.diffusion_model = TrtUnet()

    @staticmethod
    def load_lcm():
        print("Deprecated for now")
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
                 strength=0.65,
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
                 enable_prompt_blend=False,
                 use_areas= False,
                 areas= None,
                 *args,
                 **kwargs):

        if self.pipeline_type == "comfy":
            if seed == -1:
                seed = secrets.randbelow(18446744073709551615)

            if strength <= 0.0 or strength >= 1.0:
                strength = 1.0
                reset_noise = True
                init_image = None
            if subseed == -1:
                subseed = secrets.randbelow(18446744073709551615)

            if cnet_image is not None:
                cnet_image = torch.from_numpy(np.array(cnet_image).astype(np.float32) / 255.0).unsqueeze(0)

            if init_image is None or reset_noise:
                print(reset_noise, strength)
                strength = 1.0
                if latent is None:

                    if width is None:
                        width = 1024
                    if height is None:
                        height = 960
                    latent = self.generate_latent(width, height, seed, subseed, subseed_strength, seed_resize_from_h,
                                                  seed_resize_from_w, reset_noise)
                else:
                    if isinstance(latent, torch.Tensor):
                        latent = {"samples":latent}
                    elif isinstance(latent, list):
                        latent = {"samples":torch.stack(latent, dim=0)}
                    else:
                        latent = latent
            else:

                latent = torch.from_numpy(np.array(init_image).astype(np.float32) / 255.0).unsqueeze(0)
                latent = self.encode_latent(latent)
            assert isinstance(latent, dict), \
                "Our Latents have to be in a dict format with the latent being the 'samples' value"

            cond = []

            if pooled_prompts is None:
                cond = self.get_conds(prompt)
            elif pooled_prompts is not None:
                cond = pooled_prompts

            if use_areas and areas is not None:
                from nodes import ConditioningSetArea
                area_setter = ConditioningSetArea()
                for area in areas:
                    print("AREA TO USE", area)
                    prompt = area.get("prompt", None)
                    if prompt:

                        new_cond = self.get_conds(area["prompt"])
                        new_cond = area_setter.append(conditioning=new_cond, width=int(area["w"]), height=int(area["h"]), x=int(area["x"]),
                                                      y=int(area["y"]), strength=area["s"])[0]
                        cond += new_cond

            self.n_cond = self.get_conds(negative_prompt)
            self.prompt = prompt


            if next_prompt is not None and enable_prompt_blend:
                if next_prompt != prompt and next_prompt != "":
                    if 0.0 < prompt_blend < 1.0:
                        next_cond = self.get_conds(next_prompt)

                        cond = blend_tensors(cond[0], next_cond[0], blend_value=prompt_blend)

            if cnet_image is not None:
                cond = apply_controlnet(cond, self.controlnet, cnet_image, 1.0)

            # from nodes import common_ksampler as ksampler

            last_step = int((strength) * steps) if (strength != 1.0 or not reset_noise) else steps
            # last_step = steps if last_step is None else last_step
            last_step = steps
            sample = common_ksampler_with_custom_noise(model=self.model,
                                                       seed=seed,
                                                       steps=steps,
                                                       cfg=scale,
                                                       sampler_name=sampler_name,
                                                       scheduler=scheduler,
                                                       positive=cond,
                                                       negative=self.n_cond,
                                                       latent=latent,
                                                       denoise=strength,
                                                       disable_noise=False,
                                                       start_step=0,
                                                       last_step=last_step,
                                                       force_full_denoise=True,
                                                       noise=self.rng)


            if sample[0]["samples"].shape[0] == 1:
                decoded = self.decode_sample(sample[0]["samples"])
                np_array = np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0]
                image = Image.fromarray(np_array)
                # image = Image.fromarray(np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0])
                image = image.convert("RGB")
                if return_latent:
                    return sample[0]["samples"], image
                else:
                    return image
            else:
                print("decoding multi images")
                images = []
                x_samples = self.vae.decode_tiled(sample[0]["samples"])
                for sample in x_samples:
                    np_array = np.clip(255. * sample.cpu().numpy(), 0, 255).astype(np.uint8)
                    image = Image.fromarray(np_array)
                    # image = Image.fromarray(np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0])
                    image = image.convert("RGB")
                    images.append(image)


                return images

        elif self.pipeline_type == "diffusers_lcm":
            if init_image is None:
                image = self.pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    guidance_scale=scale,
                    num_inference_steps=int(steps / 5),
                    num_images_per_prompt=1,
                    lcm_origin_steps=50,
                    output_type="pil",
                ).images[0]
            else:
                # init_image = np.array(init_image)
                # init_image = Image.fromarray(init_image)
                image = self.img2img_pipe(
                    prompt=prompt,
                    strength=strength,
                    image=init_image,
                    width=width,
                    height=height,
                    guidance_scale=scale,
                    num_inference_steps=int(steps / 5),
                    num_images_per_prompt=1,
                    lcm_origin_steps=50,
                    output_type="pil",
                ).images[0]

            return image

    def decode_sample(self, sample):
        with torch.inference_mode():
            sample = sample.to(torch.float32)
            self.vae.first_stage_model.cuda()
            decoded = self.vae.decode_tiled(sample).detach()

        return decoded


def common_ksampler_with_custom_noise(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,
                                      denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                                      force_full_denoise=False, noise=None):
    latent_image = latent["samples"]
    if noise is not None:
        rng_noise = noise.next().detach().cpu()
        noise = rng_noise.clone()
    else:
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
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

    samples = sample_k(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                       denoise=denoise, disable_noise=disable_noise, start_step=start_step,
                       last_step=last_step,
                       force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=None,
                       disable_pbar=False, seed=seed)
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
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = True
            c.append(n)
    return c
