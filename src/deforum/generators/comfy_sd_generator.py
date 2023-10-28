import secrets

import numpy as np
import torch
from PIL import Image

from .rng_noise_generator import ImageRNGNoise
from ..utils.deforum_cond_utils import blend_tensors


class ComfyDeforumGenerator:

    def __init__(self, model_path:str=None, lcm=False, trt=False):
        # from deforum.datafunctions.ensure_comfy import ensure_comfy
        # ensure_comfy()
        # from deforum.datafunctions import comfy_functions
        #from comfy import model_management, controlnet

        #model_management.vram_state = model_management.vram_state.HIGH_VRAM
        self.clip_skip = -2
        self.device = "cuda"

        self.prompt = ""
        self.n_prompt = ""

        self.cond = None
        self.n_cond = None

        if not lcm:
            # if model_path == None:
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
    def encode_latent(self, latent, subseed, subseed_strength):
        with torch.inference_mode():
            latent = latent.to(torch.float32)
            latent = self.vae.encode_tiled(latent[:,:,:,:3])
            latent = latent.to("cuda")

        return {"samples":latent}

    def generate_latent(self, width, height, seed, subseed, subseed_strength, seed_resize_from_h=None, seed_resize_from_w=None, reset_noise=False):
        shape = [4, height // 8, width // 8]
        if self.rng == None or reset_noise:
            self.rng = ImageRNGNoise(shape=shape, seeds=[seed], subseeds=[subseed], subseed_strength=subseed_strength, seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
        noise = self.rng.next()
        # noise = torch.zeros([1, 4, width // 8, height // 8])
        return {"samples":noise}

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
    def load_model(self, model_path:str, trt:bool=False):

        class DummyModel:
            def __init__(self, model):
                self.model = model
        # comfy.sd.load_checkpoint_guess_config
        # model_path = "/home/mix/Downloads/SSD-1B.safetensors"
        import comfy.sd
        self.model, self.clip, self.vae, clipvision = comfy.sd.load_checkpoint_guess_config(model_path, output_vae=True,
                                                                             output_clip=True,
                                                                             embedding_directory="models/embeddings",
                                                                             output_clipvision=False,
                                                                             )

        if trt:
            from deforum.datafunctions.enable_comfy_trt import TrtUnet
            self.model.model.diffusion_model = TrtUnet()

    def load_lcm(self):
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
                 *args,
                 **kwargs):

        SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
        if self.pipeline_type == "comfy":
            if seed == -1:
                seed = secrets.randbelow(18446744073709551615)

            if strength > 1:
                strength = 1.0
                init_image = None
            if strength == 0.0:
                strength = 1.0
            if subseed == -1:
                subseed = secrets.randbelow(18446744073709551615)

            if cnet_image is not None:
                cnet_image = torch.from_numpy(np.array(cnet_image).astype(np.float32) / 255.0).unsqueeze(0)

            if init_image is None:
                if width == None:
                    width = 1024
                if height == None:
                    height = 960
                latent = self.generate_latent(width, height, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, reset_noise)

            else:
                latent = torch.from_numpy(np.array(init_image).astype(np.float32) / 255.0).unsqueeze(0)

                latent = self.encode_latent(latent, subseed, subseed_strength)


            if self.prompt != prompt or self.cond == None:
                if prompt is not None:
                    self.cond = self.get_conds(prompt)

                    # print(self.cond[0][0].shape)
                    # print(self.cond[0][1]["pooled_output"].shape)

                    self.n_cond = self.get_conds(negative_prompt)
                    self.prompt = prompt

            if next_prompt is not None:
                if next_prompt != prompt and next_prompt != "":
                    if 0.0 < prompt_blend < 1.0:
                        next_cond = self.get_conds(next_prompt)

                        self.cond = blend_tensors(self.cond[0], next_cond[0], blend_value=prompt_blend)



            if cnet_image is not None:
                self.cond = apply_controlnet(cond, self.controlnet, cnet_image, 1.0)


            # from nodes import common_ksampler as ksampler

            last_step = int((1-strength) * steps) + 1 if strength != 1.0 else steps
            last_step = steps if last_step == None else last_step

            sample = common_ksampler_with_custom_noise(model=self.model,
                                                       seed=seed,
                                                       steps=steps,
                                                       cfg=scale,
                                                       sampler_name=sampler_name,
                                                       scheduler=scheduler,
                                                       positive=self.cond,
                                                       negative=self.n_cond,
                                                       latent=latent,
                                                       denoise=strength,
                                                       disable_noise=False,
                                                       start_step=0,
                                                       last_step=last_step,
                                                       force_full_denoise=True,
                                                       noise=self.rng)

            decoded = self.decode_sample(sample[0]["samples"])

            np_array = np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0]
            image = Image.fromarray(np_array)
            #image = Image.fromarray(np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0])
            image = image.convert("RGB")
            if return_latent:
                return sample[0]["samples"], image
            else:
                return image
        elif self.pipeline_type == "diffusers_lcm":
            if init_image is None:
                image = self.pipe(
                        prompt=prompt,
                        width=width,
                        height=height,
                        guidance_scale=scale,
                        num_inference_steps=int(steps/5),
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
                        num_inference_steps=int(steps/5),
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
            return (conditioning, )

        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = True
            c.append(n)
    return c
#
#
# def replace_forward_with(control_net_model, new_forward):
#     def forward_with_self(*args, **kwargs):
#         return new_forward(control_net_model, *args, **kwargs)
#     return forward_with_self
#
# def apply_model_for_trt(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
#     if c_concat is not None:
#         xc = torch.cat([x] + [c_concat], dim=1)
#     else:
#         xc = x
#     context = c_crossattn
#     dtype = self.get_dtype()
#     xc = xc.to(dtype)
#     t = t.to(dtype)
#     context = context.to(dtype)
#     extra_conds = {}
#     for o in kwargs:
#         extra_conds[o] = kwargs[o].to(dtype)
#     return self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options,
#                                 **extra_conds).float()
#
#
# def forward_for_trt(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
#     """
#     Apply the model to an input batch.
#     :param x: an [N x C x ...] Tensor of inputs.
#     :param timesteps: a 1-D batch of timesteps.
#     :param context: conditioning plugged in via crossattn
#     :param y: an [N] Tensor of labels, if class-conditional.
#     :return: an [N x C x ...] Tensor of outputs.
#     """
#
#     # print("our forward")
#     transformer_options["original_shape"] = list(x.shape)
#     transformer_options["current_index"] = 0
#     transformer_patches = transformer_options.get("patches", {})
#
#     assert (y is not None) == (
#         self.num_classes is not None
#     ), "must specify y if and only if the model is class-conditional"
#     hs = []
#     from src.ComfyUI.comfy.ldm.modules.diffusionmodules.util import timestep_embedding
#     t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
#     emb = self.time_embed(t_emb)
#
#     if self.num_classes is not None:
#         assert y.shape[0] == x.shape[0]
#         emb = emb + self.label_emb(y)
#
#     h = x.type(self.dtype)
#     for id, module in enumerate(self.input_blocks):
#         transformer_options["block"] = ("input", id)
#         from src.ComfyUI.comfy.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed
#         h = forward_timestep_embed(module, h, emb, context, transformer_options)
#         if control is not None and 'input' in control and len(control['input']) > 0:
#             ctrl = control['input'].pop()
#             if ctrl is not None:
#                 h += ctrl
#         hs.append(h)
#     transformer_options["block"] = ("middle", 0)
#     h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options)
#     if control is not None and 'middle' in control and len(control['middle']) > 0:
#         ctrl = control['middle'].pop()
#         if ctrl is not None:
#             h += ctrl
#
#     for id, module in enumerate(self.output_blocks):
#         transformer_options["block"] = ("output", id)
#         hsp = hs.pop()
#         if control is not None and 'output' in control and len(control['output']) > 0:
#             ctrl = control['output'].pop()
#             if ctrl is not None:
#                 hsp += ctrl
#
#         if "output_block_patch" in transformer_patches:
#             patch = transformer_patches["output_block_patch"]
#             for p in patch:
#                 h, hsp = p(h, hsp, transformer_options)
#
#         h = torch.cat([h, hsp], dim=1)
#         del hsp
#         if len(hs) > 0:
#             output_shape = hs[-1].shape
#         else:
#             output_shape = None
#         h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape)
#     h = h.type(x.dtype)
#     if self.predict_codebook_ids:
#         return self.id_predictor(h)
#     else:
#         return self.out(h)