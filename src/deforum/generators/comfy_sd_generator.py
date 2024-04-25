import secrets

import numpy as np
import torch
from PIL import Image

from .comfy_utils import ensure_comfy
from .rng_noise_generator import ImageRNGNoise, slerp
from deforum.utils.deforum_cond_utils import blend_tensors
from deforum.utils.logging_config import logger

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
        self.loaded_lora = None
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

    def encode_latent(self, vae, latent, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, reset_noise=False):

        ## TODO this looks wrong! Why override the supplied subseed strength?
        #subseed_strength = 0.6

        with torch.inference_mode():
            latent = latent.to(torch.float32)
            latent = vae.encode_tiled(latent[:, :, :, :3])
            latent = latent.to("cuda")
        if self.rng is None or reset_noise:
            self.rng = ImageRNGNoise(shape=latent[0].shape, seeds=[seed], subseeds=[subseed], subseed_strength=subseed_strength,
                                     seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
        #     noise = self.rng.first()
        # #     noise = slerp(subseed_strength, noise, latent)
        # else:
        #     noise = self.rng.next()
        #     noise = slerp(subseed_strength, noise, latent)
        #     noise = latent
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

    def get_conds(self, clip, prompt):
        with torch.inference_mode():
            clip_skip = -1
            if clip_skip != clip_skip or clip.layer_idx != clip_skip:
                clip.layer_idx = clip_skip
                clip.clip_layer(clip_skip)
                self.clip_skip = clip_skip

            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]

    def load_model(self, model_path: str, trt: bool = False):
        try:
            self.cleanup()
        except:
            pass
        import comfy.sd
        model, clip, vae, clipvision = (
            comfy.sd.load_checkpoint_guess_config(model_path,
                                                  output_vae=True,
                                                  output_clip=True,
                                                  embedding_directory="models/embeddings",
                                                  output_clipvision=False,
                                                  )
        )
        # from comfy import utils
        # lora_path = "/home/mix/Downloads/pytorch_lora_weights.safetensors"
        # lora = utils.load_torch_file(lora_path, safe_load=True)
        # self.model, self.clip = comfy.sd.load_lora_for_models(
        #     self.model, self.clip, lora, 1.0, 1.0)
        trt = False
        if trt:
            from ..optimizations.deforum_comfy_trt.deforum_trt_comfyunet import TrtUnet
            model.model.diffusion_model = TrtUnet()
        return model, clip, vae, clipvision
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

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
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

        if self.pipeline_type == "comfy":
            if (self.vae is None):
                import comfy.sd
                self.model, self.clip, self.vae, self.clipvision = comfy.sd.load_checkpoint_guess_config(self.model_path,
                                                    output_vae=True,
                                                    output_clip=True,
                                                    embedding_directory="models/embeddings",
                                                    output_clipvision=False,
                                                    )

            if seed == -1:
                seed = secrets.randbelow(18446744073709551615)

            # strength = 1 - strength

            if strength <= 0.0 or strength >= 1.0:
                strength = 1.0
                reset_noise = True
                init_image = None
            if subseed == -1:
                subseed = secrets.randbelow(18446744073709551615)

            if cnet_image is not None:
                cnet_image = torch.from_numpy(np.array(cnet_image).astype(np.float32) / 255.0).unsqueeze(0)

            if init_image is None or reset_noise:
                logger.info(f"reset_noise: {reset_noise}; resetting strength to 1.0 from: {strength}")
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
                latent = self.encode_latent(self.vae, latent, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w)
            assert isinstance(latent, dict), \
                "Our Latents have to be in a dict format with the latent being the 'samples' value"

            cond = []

            if pooled_prompts is None and prompt is not None:
                cond = self.get_conds(self.clip, prompt)
            elif pooled_prompts is not None:
                cond = pooled_prompts

            if use_areas and areas is not None:
                from nodes import ConditioningSetArea
                area_setter = ConditioningSetArea()
                for area in areas:
                    logger.info(f"AREA TO USE: {area}")
                    prompt = area.get("prompt", None)
                    if prompt:

                        new_cond = self.get_conds(self.clip, area["prompt"])
                        new_cond = area_setter.append(conditioning=new_cond, width=int(area["w"]), height=int(area["h"]), x=int(area["x"]),
                                                      y=int(area["y"]), strength=area["s"])[0]
                        cond += new_cond

            self.n_cond = self.get_conds(self.clip, negative_prompt)
            self.prompt = prompt


            if next_prompt is not None and enable_prompt_blend:
                if next_prompt != prompt and next_prompt != "":
                    if 0.0 < prompt_blend < 1.0:
                        next_cond = self.get_conds(self.clip, next_prompt)

                        cond = blend_tensors(cond[0], next_cond[0], blend_value=prompt_blend)

            if cnet_image is not None:
                cond = apply_controlnet(cond, self.controlnet, cnet_image, 1.0)

            from nodes import common_ksampler as ksampler

            #steps = int((strength) * steps) if (strength != 1.0 or not reset_noise) else steps
            last_step = steps# if last_step is None else last_step

            logger.info(f"seed/subseed/subseed_str={seed}/{subseed}/{subseed_strength}; strength={strength}; scale={scale}; sampler_name={sampler_name}; scheduler={scheduler};")

            # denoise = 1-strength
            steps = int(strength * steps)
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
                                                       force_full_denoise=True, # TODO - what does this do?
                                                       noise=self.rng)



            if sample[0]["samples"].shape[0] == 1:
                decoded = self.decode_sample(self.vae, sample[0]["samples"])
                np_array = np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0]
                image = Image.fromarray(np_array)
                # image = Image.fromarray(np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0])
                image = image.convert("RGB")

                image.save('test.png', "PNG")

                if return_latent:
                    return sample[0]["samples"], image
                else:
                    return image
            else:
                logger.info("decoding multi images")
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

    def decode_sample(self, vae, sample):
        with torch.inference_mode():
            sample = sample.to(torch.float32)
            vae.first_stage_model.cuda()
            decoded = vae.decode_tiled(sample).detach()

        return decoded

    def cleanup(self):
        return
        self.model.unpatch_model(device_to='cpu')
        self.vae.first_stage_model.to('cpu')
        #self.clip.to('cpu')
        del self.model
        del self.vae
        del self.clip


def common_ksampler_with_custom_noise(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,
                                      denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                                      force_full_denoise=False, noise=None):
    latent_image = latent["samples"]
    if noise is not None:
        noise = noise.next()#.detach().cpu()
        # noise = rng_noise.clone()
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
