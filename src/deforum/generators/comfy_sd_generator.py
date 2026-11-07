"""
comfy_sd_generator.py

ComfyDeforumGenerator: the image generator behind every Deforum animation mode.

It drives ComfyUI's core model code directly, so any model family ComfyUI can load works:

* single-file checkpoints (SD 1.x / 2.x / SDXL / SD3 / FLUX all-in-one files) -> ``model_path``
* split "diffusion model + text encoder(s) + VAE" releases (FLUX.1, FLUX.2, FLUX.3, Chroma,
  Qwen-Image, ...) -> ``model_path`` points at the diffusion model; text encoders and VAE are
  picked automatically from the ComfyUI model folders, or given explicitly with
  ``text_encoders=[...]``, ``vae_path=...`` and ``clip_type=...``.

Latent shapes (channels, spatial downscale, 2D/3D latents) are taken from the loaded model, so
SD's 4ch/8x, FLUX.1's 16ch/8x and FLUX.2/3's 128ch/16x latents are all handled the same way.
Noise uses Deforum's A1111-compatible RNG, so seeds and sub-seed variations behave as before.
"""
import os
import secrets
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from deforum.utils.logging_config import logger

from ..utils.constants import config
from ..utils.device import empty_cache, get_device
from ..utils.model_download import download_from_civitai_by_version_id
from .comfy_utils import ensure_comfy
from .rng_noise_generator import ImageRNGNoise


def pil2tensor(image):
    """PIL image -> [1, H, W, C] float tensor in 0..1 (ComfyUI's IMAGE layout)."""
    return torch.from_numpy(np.array(image.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)


# --------------------------------------------------------------------------------------------
# Model families. Keys are ComfyUI supported_models class names. Anything not listed uses plain
# CFG sampling and the text encoder / VAE inside the checkpoint.
#   clip_type : ComfyUI CLIPType name for loading separate text encoders
#   te        : groups of filename hints; one file is picked per group, in order
#   vae       : filename hints for the VAE (each candidate is verified against the model)
#   flow      : flow-matching model (karras/exponential schedules get swapped for a fitting one)
#   scheduler : schedule used instead of karras/exponential on flow models
# --------------------------------------------------------------------------------------------
_FLUX1_TE = [["clip_l"], ["t5xxl", "t5-xxl", "t5_xxl", "t5"]]
_FLUX1_VAE = ["flux1", "ae.sft", "flux_vae", "ae.safetensors"]
FAMILIES = {
    "SD15": {"label": "SD 1.5"},
    "SD20": {"label": "SD 2.x"},
    "SDXL": {"label": "SDXL"},
    "SDXLRefiner": {"label": "SDXL Refiner"},
    "SSD1B": {"label": "SSD-1B"},
    "SD3": {"label": "SD3", "clip_type": "sd3", "flow": True,
            "te": [["clip_l"], ["clip_g"], ["t5xxl", "t5-xxl", "t5_xxl"]], "vae": ["sd3", "sd35"]},
    "Flux": {"label": "FLUX.1", "clip_type": "flux", "flow": True, "te": _FLUX1_TE, "vae": _FLUX1_VAE},
    "FluxSchnell": {"label": "FLUX.1 schnell", "clip_type": "flux", "flow": True, "te": _FLUX1_TE, "vae": _FLUX1_VAE},
    "Chroma": {"label": "Chroma", "clip_type": "chroma", "flow": True, "te": _FLUX1_TE[1:], "vae": _FLUX1_VAE},
    "Flux2": {"label": "FLUX.2", "clip_type": "flux2", "flow": True, "scheduler": "flux2",
              "te": [["mistral_3_small_flux2", "mistral3", "mistral"]],  # Klein: see _text_encoder_groups
              "vae": ["flux2-vae", "flux2_vae", "flux2", "ae.safetensors"]},
    # BFL's reference workflow: euler / simple, 16 steps, CFG 1, constant flow shift ln(16)
    "Flux3": {"label": "FLUX.3", "clip_type": "flux3", "flow": True, "scheduler": "simple", "flow_shift": 2.77259,
              "te": [["qwen3vl_4b", "qwen3-vl-4b", "qwen3_vl_4b"]],
              "vae": ["flux2-vae", "flux2_vae", "flux3", "flux2", "ae.safetensors"]},
    "QwenImage": {"label": "Qwen-Image", "clip_type": "qwen_image", "flow": True,
                  "te": [["qwen_2.5_vl", "qwen2.5_vl", "qwen_2_5_vl", "qwen2_5_vl"]], "vae": ["qwen_image_vae", "qwen_image"]},
    "Lumina2": {"label": "Lumina 2 / Z-Image", "clip_type": "lumina2", "flow": True,
                "te": [["qwen_3_4b", "qwen3_4b", "gemma_2_2b", "gemma2"]], "vae": _FLUX1_VAE},
}

FLOW_DEFAULT_SCHEDULER = "simple"
ETA_NOISE_SEED_DELTA = 31337
_SCHEDULERS_UNFIT_FOR_FLOW = {"karras", "exponential", "polyexponential"}
_MODEL_EXTS = {".safetensors", ".sft", ".gguf", ".ckpt", ".pt", ".bin"}


def _find_model_file(kind, hints, exclude=()):
    """First file in ComfyUI's ``kind`` folders whose name contains one of ``hints`` (hint order wins)."""
    import folder_paths
    names = sorted((n for n in folder_paths.get_filename_list(kind) if os.path.splitext(n)[1].lower() in _MODEL_EXTS),
                   key=len)
    for hint in hints:
        for name in names:
            path = folder_paths.get_full_path(kind, name)
            if hint.lower() in os.path.basename(name).lower() and path not in exclude:
                return path
    return None


def _resolve_model_path(kind, name_or_path):
    if name_or_path is None or os.path.isfile(name_or_path):
        return name_or_path
    import folder_paths
    path = folder_paths.get_full_path(kind, name_or_path)
    if path is None:
        raise FileNotFoundError(f"{kind} model '{name_or_path}' not found in {folder_paths.get_folder_paths(kind)}")
    return path


def _pad_tokens(a, b):
    """Zero-pad the token dimension of two cond tensors to the same length (T5/LLM encoders vary)."""
    n = max(a.shape[1], b.shape[1])

    def pad(t):
        if t.shape[1] == n:
            return t
        return torch.cat([t, t.new_zeros((t.shape[0], n - t.shape[1], *t.shape[2:]))], 1)

    return pad(a), pad(b)


def blend_conditioning(cond_from, cond_to, t):
    """Linear blend between two ComfyUI conditionings; t=0 -> cond_from, t=1 -> cond_to."""
    out = []
    for i, (c_from, extra_from) in enumerate(cond_from):
        c_to, extra_to = cond_to[min(i, len(cond_to) - 1)]
        a, b = _pad_tokens(c_from, c_to.to(c_from.device, c_from.dtype))
        extra = extra_from.copy()
        pa, pb = extra_from.get("pooled_output"), extra_to.get("pooled_output")
        if pa is not None and pb is not None and pa.shape == pb.shape:
            extra["pooled_output"] = torch.lerp(pa, pb.to(pa), t)
        extra.pop("attention_mask", None)  # padded positions are real (zero) tokens now
        out.append([torch.lerp(a, b, t), extra])
    return out


class ComfyDeforumGenerator:
    """
    Generates single frames (txt2img / img2img) for the Deforum animation pipeline, for any model
    family ComfyUI supports.

    Args:
        model_path: checkpoint or diffusion-model file (path, or a name inside the model folders).
        text_encoders: explicit text encoder files for split models (otherwise auto-detected).
        vae_path: explicit VAE file (otherwise the checkpoint's own, or auto-detected).
        clip_type: ComfyUI CLIPType name ("flux", "flux2", "flux3", "sd3", ...) for text_encoders.
    """

    def __init__(self, model_path: str = None, text_encoders=None, vae_path=None, clip_type=None,
                 *args, **kwargs):
        ensure_comfy()
        self.model_path = model_path
        self.text_encoder_paths = list(text_encoders) if text_encoders else None
        self.vae_path = vae_path
        self.clip_type = clip_type
        self.model = self.base_model = None
        self.clip = self.base_clip = None
        self.vae = None
        self.family = None
        self.model_loaded = False
        self.loaded_lora = None
        self.ip_adapter = None
        self.clip_vision = None
        self.controlnet = None
        self.optimize = False
        self.optimized = False
        self.current_optimization = ""
        self.optimizations = []
        self._cond_cache = OrderedDict()
        self._pinned_conds = set()
        self.prompt = self.n_prompt = ""
        self.cond = self.n_cond = None

    # ---------------------------------------------------------------- properties
    @property
    def device(self):
        return get_device()

    @property
    def family_info(self):
        return FAMILIES.get(self.family, {})

    @property
    def latent_format(self):
        return self.model.get_model_object("latent_format")

    @property
    def downscale(self):
        return int(self.latent_format.spacial_downscale_ratio)

    @property
    def uses_guidance_embed(self):
        """True for guidance-distilled models (FLUX.1-dev, FLUX.2-dev) that take guidance instead of CFG."""
        params = getattr(getattr(self.model.model, "diffusion_model", None), "params", None)
        return bool(getattr(params, "guidance_embed", False))

    def describe(self):
        if not self.model_loaded:
            return "not loaded"
        lf = self.latent_format
        return (f"{self.family_info.get('label', self.family)} [{self.family}], latent {lf.latent_channels}ch "
                f"/{self.downscale}x, guidance-embed={self.uses_guidance_embed}, device={self.device}")

    # ---------------------------------------------------------------- loading
    def optimize_model(self, optimization: str = None):
        """Kept for API compatibility. Stable-Fast / OneDiff are unmaintained and no longer offered."""
        if optimization:
            logger.debug(f"Optimization '{optimization}' is not available, running unoptimized.")

    def set_components(self, model_path=None, text_encoders=None, vae_path=None, clip_type=None):
        """Point the generator at different model files; they load lazily on the next frame."""
        changed = (bool(model_path) and model_path != self.model_path) or \
                  (text_encoders is not None and list(text_encoders) != (self.text_encoder_paths or [])) or \
                  (vae_path is not None and (vae_path or None) != self.vae_path) or \
                  (clip_type is not None and (clip_type or None) != self.clip_type)
        if not changed:
            return False
        self.cleanup()
        if model_path:
            self.model_path = model_path
        if text_encoders is not None:
            self.text_encoder_paths = list(text_encoders) or None
        if vae_path is not None:
            self.vae_path = vae_path or None
        if clip_type is not None:
            self.clip_type = clip_type or None
        return True

    def load_model(self):
        """Load the diffusion model, text encoder(s) and VAE (no-op when already loaded)."""
        if self.model_loaded:
            return
        import comfy.sd
        import folder_paths

        if not self.model_path:
            raise ValueError("No model selected: pass model_path (a checkpoint or diffusion model file).")
        path = self.model_path
        if not os.path.isfile(path):
            path = folder_paths.get_full_path("checkpoints", path) or folder_paths.get_full_path("diffusion_models", path)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model_path = path
        embeddings = folder_paths.get_folder_paths("embeddings")
        logger.info(f"Loading model {path}")

        model = clip = vae = None
        try:
            model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
                path, output_vae=True, output_clip=True, embedding_directory=embeddings)
        except RuntimeError as e:
            # Diffusion-model-only files (FLUX.2/3 and most modern releases) are not full checkpoints
            logger.debug(f"Not a full checkpoint ({e}); loading as a diffusion model")
        if model is None:
            model = comfy.sd.load_diffusion_model(path)
        if vae is not None:
            try:  # files without VAE weights still yield an (invalid) VAE object
                vae.throw_exception_if_invalid()
            except Exception:
                vae = None
        if clip is not None and getattr(clip, "cond_stage_model", None) is None:
            clip = None
        self.family = type(model.model.model_config).__name__
        shift = self.family_info.get("flow_shift")
        if shift:  # constant flow shift, as ComfyUI's ModelSamplingFlux node applies it
            from comfy_extras.nodes_model_advanced import ModelSamplingFlux
            model = ModelSamplingFlux().patch(model, shift, shift, 1024, 1024)[0]
        self.model = self.base_model = model
        label = self.family_info.get("label", self.family)

        if self.text_encoder_paths:
            clip = self._load_text_encoders(self.text_encoder_paths, embeddings)
        elif clip is None:
            clip = self._load_text_encoders(self._auto_text_encoders(), embeddings)
        if self.vae_path:
            vae = self._load_vae(_resolve_model_path("vae", self.vae_path))
        elif vae is None:
            vae = self._auto_vae()
        if vae is None:
            raise RuntimeError(f"No VAE found for {label}: put one in the vae folder or set vae_path.")
        self.clip = self.base_clip = clip
        self.vae = vae
        self.model_loaded = True
        self._cond_cache.clear()
        self._pinned_conds = set()
        self.cond = self.n_cond = None
        logger.info(f"Model ready: {self.describe()}")

    # FLUX.2 dev and FLUX.2 Klein share a model class; the text-input width tells their encoders apart
    _FLUX2_TE_BY_CONTEXT = {15360: ["mistral_3_small_flux2", "mistral3", "mistral"],
                            7680: ["qwen_3_4b", "qwen3_4b", "qwen3-4b"],
                            12288: ["qwen_3_8b", "qwen3_8b", "qwen3-8b"]}

    def _text_encoder_groups(self):
        groups = self.family_info.get("te")
        if self.family == "Flux2":
            params = getattr(self.model.model.diffusion_model, "params", None)
            hints = self._FLUX2_TE_BY_CONTEXT.get(getattr(params, "context_in_dim", None))
            if hints:
                groups = [hints]
        return groups

    def _auto_text_encoders(self):
        groups = self._text_encoder_groups()
        if not groups:
            raise RuntimeError(
                f"'{os.path.basename(self.model_path)}' ({self.family}) has no built-in text encoder and no "
                f"auto-detection recipe; pass text_encoders=[...] and clip_type.")
        found = []
        for hints in groups:
            p = _find_model_file("text_encoders", hints, exclude=found)
            if p is None:
                if self.family == "SD3" and "t5xxl" in hints:
                    continue  # SD3 also runs without T5
                raise RuntimeError(f"Missing text encoder for {self.family}: expected a file named like "
                                   f"'{hints[0]}*' in a text_encoders folder.")
            found.append(p)
        logger.info(f"Auto-selected text encoder(s): {[os.path.basename(p) for p in found]}")
        return found

    def _load_text_encoders(self, paths, embeddings):
        import comfy.sd
        paths = [_resolve_model_path("text_encoders", p) for p in paths]
        clip_type_name = (self.clip_type or self.family_info.get("clip_type") or "stable_diffusion").upper()
        clip_type = getattr(comfy.sd.CLIPType, clip_type_name, comfy.sd.CLIPType.STABLE_DIFFUSION)
        return comfy.sd.load_clip(ckpt_paths=paths, embedding_directory=embeddings, clip_type=clip_type)

    @staticmethod
    def _load_vae(path):
        import comfy.sd
        import comfy.utils
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    def _vae_matches(self, vae):
        """A VAE fits when a tiny encode yields the model's latent channel count."""
        try:
            size = self.downscale * 4
            lat = vae.encode(torch.zeros((1, size, size, 3)))
            return lat.shape[1] == self.latent_format.latent_channels
        except Exception as e:
            logger.debug(f"VAE probe failed: {e}")
            return False

    def _auto_vae(self):
        import folder_paths
        candidates = []
        for hint in self.family_info.get("vae", []):
            p = _find_model_file("vae", [hint])
            if p and p not in candidates:
                candidates.append(p)
        for name in folder_paths.get_filename_list("vae"):  # then any other image-sized VAE
            p = folder_paths.get_full_path("vae", name)
            if p and p not in candidates and os.path.splitext(p)[1].lower() in _MODEL_EXTS \
                    and os.path.getsize(p) < 1_500_000_000:
                candidates.append(p)
        for p in candidates:
            try:
                vae = self._load_vae(p)
            except Exception as e:
                logger.debug(f"Skipping VAE {p}: {e}")
                continue
            if self._vae_matches(vae):
                logger.info(f"Auto-selected VAE: {os.path.basename(p)}")
                return vae
            del vae
        return None

    # ---------------------------------------------------------------- LoRA / adapters
    def load_lora_from_civitai(self, lora_id="", model_strength=0.0, clip_strength=0.0):
        cache_dir = os.path.join(config.root_path, "models")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            self.load_model()
            filename = download_from_civitai_by_version_id(model_id=lora_id, destination=cache_dir,
                                                           force_download=False)
            self.model, self.clip = self.load_lora(self.model, self.clip, os.path.join(cache_dir, filename),
                                                   model_strength, clip_strength)
            self._cond_cache.clear()
            self._pinned_conds = set()
        except Exception as e:
            logger.info(f"LoRA loading from CivitAI failed, check version id {lora_id}: {e}")

    def load_lora(self, model, clip, lora_path, strength_model, strength_clip):
        import comfy.sd
        import comfy.utils
        if strength_model == 0 and strength_clip == 0:
            return model, clip
        lora_path = _resolve_model_path("loras", lora_path)
        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
            lora = self.loaded_lora[1]
        else:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)
        return comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

    def apply_loras(self, loras):
        """loras: list of (path_or_name, model_strength, clip_strength), applied on top of the base model."""
        self.load_model()
        model, clip = self.base_model, self.base_clip
        for path, sm, sc in loras or []:
            model, clip = self.load_lora(model, clip, path, sm, sc)
        self.model, self.clip = model, clip
        self._cond_cache.clear()
        self._pinned_conds = set()

    @torch.inference_mode()
    def set_ip_adapter_image(self, image: Image.Image, weight: float = 1.0, start: float = 0.0, end: float = 1.0):
        """IP-Adapter (SD1.5/SDXL) via ComfyUI_IPAdapter_plus, when it is installed in ComfyUI's custom_nodes."""
        self.load_model()
        try:
            from .comfy_utils import load_comfy_custom_nodes
            load_comfy_custom_nodes()
            from nodes import NODE_CLASS_MAPPINGS
            apply = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
            if self.ip_adapter is None:
                import comfy.clip_vision
                loader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
                ip_name = "ip-adapter-plus_sd15.safetensors" if self.family == "SD15" else \
                    "ip-adapter-plus_sdxl_vit-h.safetensors"
                self.ip_adapter = loader.load_ipadapter_model(ip_name)[0]
                self.clip_vision = comfy.clip_vision.load(
                    _find_model_file("clip_vision", ["vit-h", "vit_h", "clip-vit-h"]))
        except Exception as e:
            logger.warning(f"IP-Adapter unavailable (needs ComfyUI_IPAdapter_plus and its models): {e}")
            return
        self.model = apply.apply_ipadapter(self.model, self.ip_adapter, clip_vision=self.clip_vision,
                                           image=pil2tensor(image), weight=weight, weight_type="linear",
                                           combine_embeds="concat", start_at=start, end_at=end,
                                           embeds_scaling="V only")[0]

    # ---------------------------------------------------------------- conditioning
    def get_conds(self, clip, prompt, *args, **kwargs):
        """Encode a prompt with ComfyUI's text encoder (cached: prompts repeat across frames).

        A1111-style weights such as ``(word:1.2)`` are understood by ComfyUI's tokenizer."""
        prompt = prompt or ""
        key = (id(clip), prompt)
        if key in self._cond_cache:
            self._cond_cache.move_to_end(key)
            return self._cond_cache[key]
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt), show_pbar=False)
        self._cond_cache[key] = cond
        while len(self._cond_cache) > max(32, len(self._pinned_conds) + 16):
            oldest = next((k for k in self._cond_cache if k not in self._pinned_conds), None)
            if oldest is None:
                break
            self._cond_cache.pop(oldest)
        return cond

    @torch.inference_mode()
    def precache_prompts(self, prompts, max_prompts=512):
        """Encode all prompts an animation will need up front, then move the text encoder off the GPU.

        Frames afterwards only run the diffusion model and VAE, so ComfyUI never has to swap the text
        encoder back in (for FLUX-class encoders that swap costs seconds and lots of VRAM)."""
        import time

        import comfy.model_management
        self.load_model()
        unique = list(dict.fromkeys(p or "" for p in prompts))
        if len(unique) > max_prompts:
            logger.warning(f"{len(unique)} distinct prompts; pre-caching the first {max_prompts}, the rest encode on demand")
            unique = unique[:max_prompts]
        todo = [p for p in unique if (id(self.clip), p) not in self._cond_cache]
        t0 = time.time()
        self._pinned_conds = {(id(self.clip), p) for p in unique}
        for p in todo:
            self.get_conds(self.clip, p)
        if todo:
            # the text encoder is done for this animation: free its VRAM for the diffusion model
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            logger.info(f"Pre-cached {len(todo)} prompt embeddings in {time.time() - t0:.1f}s "
                        f"({len(unique) - len(todo)} already cached)")

    @staticmethod
    def _set_area(cond, area):
        """Regional prompt: area x/y/w/h in pixels, s = strength (same as ComfyUI's ConditioningSetArea)."""
        import node_helpers
        x, y, w, h = (int(area[k]) for k in ("x", "y", "w", "h"))
        return node_helpers.conditioning_set_values(cond, {
            "area": (h // 8, w // 8, y // 8, x // 8),
            "strength": float(area.get("s", 1.0)),
            "set_area_to_bounds": False,
        })

    # ---------------------------------------------------------------- latents
    def _round_size(self, width, height):
        m = self.downscale
        return max(m, int(width) // m * m), max(m, int(height) // m * m)

    def empty_latent(self, width, height):
        lf = self.latent_format
        shape = [1, lf.latent_channels, height // self.downscale, width // self.downscale]
        if getattr(lf, "latent_dimensions", 2) == 3:
            shape.insert(2, 1)
        return torch.zeros(shape)

    def make_noise(self, like, seed, subseed=0, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0):
        """A1111-compatible noise shaped like ``like`` (any channel count, 2D or 3D latents)."""
        flat = like[:, :, 0] if like.ndim == 5 else like
        rng = ImageRNGNoise(shape=list(flat.shape[1:]), seeds=[seed], subseeds=[subseed],
                            subseed_strength=subseed_strength, seed_resize_from_h=seed_resize_from_h,
                            seed_resize_from_w=seed_resize_from_w, downscale=self.downscale)
        noise = rng.first().float().cpu()
        return noise.unsqueeze(2) if like.ndim == 5 else noise

    def generate_latent(self, width=1024, height=1024, seed=0, subseed=0, subseed_strength=0.1,
                        seed_resize_from_h=None, seed_resize_from_w=None, reset_noise=False):
        """Noise for an empty latent of this size, as {"samples": tensor} (API compatibility)."""
        self.load_model()
        width, height = self._round_size(width, height)
        return {"samples": self.make_noise(self.empty_latent(width, height), seed, subseed, subseed_strength,
                                           seed_resize_from_h or 0, seed_resize_from_w or 0)}

    @torch.inference_mode()
    def encode_latent(self, vae, image):
        """[1, H, W, C] 0..1 tensor or PIL image -> {"samples": latent}."""
        if isinstance(image, Image.Image):
            image = pil2tensor(image)
        return {"samples": vae.encode(image[..., :3])}

    @torch.inference_mode()
    def decode_sample(self, vae, sample):
        """latent -> [1, H, W, C] 0..1 tensor."""
        images = vae.decode(sample)
        if images.ndim == 5:  # video VAEs return [B, T, H, W, C]
            images = images[:, 0]
        return images.clamp(0.0, 1.0)

    # ---------------------------------------------------------------- sampling
    @staticmethod
    def _custom_sigmas(scheduler, steps, denoise, width, height):
        """Schedules ComfyUI only exposes as nodes: the resolution-aware FLUX.2/3 schedule ("flux2")."""
        if scheduler != "flux2":
            return None
        from comfy_extras.nodes_flux import get_schedule
        total = steps if denoise >= 0.9999 else max(steps, int(steps / denoise))
        sigmas = torch.as_tensor(get_schedule(total, round(width * height / 256)), dtype=torch.float32)
        return sigmas[-(steps + 1):]

    def _pick_scheduler(self, scheduler):
        info = self.family_info
        if info.get("flow") and (not scheduler or scheduler in _SCHEDULERS_UNFIT_FOR_FLOW):
            new = info.get("scheduler", FLOW_DEFAULT_SCHEDULER)
            logger.debug(f"'{scheduler}' schedule does not suit {info.get('label')}; using '{new}'")
            return new
        return scheduler or "normal"

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
            guidance=None,
            *args,
            **kwargs,
    ):
        """
        Generate one frame and return it as a PIL image (plus the latent if ``return_latent``).

        ``strength`` is Deforum's img2img strength: how much of ``init_image`` survives
        (0 = regenerate fully, 1 = keep). Without an init image, or with ``reset_noise``, a txt2img
        frame is made. ``scale`` is the CFG scale; on guidance-distilled models (FLUX.1-dev,
        FLUX.2-dev) it is passed as the distilled guidance instead (override with ``guidance``).
        """
        import comfy.sample
        import comfy.samplers

        self.load_model()
        if use_optimization:
            self.optimize_model(use_optimization)

        if seed is None or int(seed) == -1:
            seed = secrets.randbelow(2 ** 63)
        if subseed is None or int(subseed) == -1:
            subseed = secrets.randbelow(2 ** 63)
        seed, subseed = int(seed) % (2 ** 64), int(subseed) % (2 ** 64)

        if init_image is not None and not isinstance(init_image, Image.Image):
            init_image = Image.fromarray(np.asarray(init_image).astype(np.uint8))
        if width is None or height is None:
            width, height = init_image.size if init_image is not None else (1024, 1024)
        width, height = self._round_size(width, height)

        denoise = 1.0 - strength if 0 < strength < 1 else 1.0
        if init_image is None or reset_noise:
            denoise = 1.0
        if denoise >= 1.0:
            sub_strength = subseed_strength
            if latent is not None:
                latent_image = torch.zeros_like(latent["samples"] if isinstance(latent, dict) else latent)
            else:
                latent_image = self.empty_latent(width, height)
        else:
            sub_strength = 0.0
            if init_image.size != (width, height):
                init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)
            latent_image = self.encode_latent(self.vae, init_image)["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(self.model, latent_image, self.downscale)

        # ---- conditioning
        if use_areas and areas:
            positive = []
            for area in areas:
                if area.get("prompt"):
                    positive += self._set_area(self.get_conds(self.clip, area["prompt"]), area)
            positive = positive or self.get_conds(self.clip, prompt or "")
        elif pooled_prompts is not None:
            positive = pooled_prompts
        elif cond is not None:
            positive = cond
        else:
            positive = self.get_conds(self.clip, prompt)
        if (enable_prompt_blend and next_prompt and next_prompt != prompt and prompt_blend is not None
                and 0.0 < prompt_blend < 1.0 and not use_areas):
            logger.debug(f"[prompt blend {prompt_blend:.3f}] -> {next_prompt[:60]}")
            positive = blend_conditioning(positive, self.get_conds(self.clip, next_prompt), float(prompt_blend))
        negative = n_cond if n_cond is not None else self.get_conds(self.clip, negative_prompt)
        self.prompt, self.n_prompt = prompt, negative_prompt
        self.cond, self.n_cond = positive, negative

        cfg = float(scale)
        if self.uses_guidance_embed:
            import node_helpers
            positive = node_helpers.conditioning_set_values(
                positive, {"guidance": float(guidance) if guidance is not None else cfg})
            cfg = 1.0
        if cnet_image is not None and self.controlnet is not None:
            positive = apply_controlnet(positive, self.controlnet, pil2tensor(cnet_image), 1.0)

        # ---- sampling
        steps = max(1, round(denoise * int(steps)))
        if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
            logger.warning(f"Unknown sampler '{sampler_name}', using euler")
            sampler_name = "euler"
        scheduler = self._pick_scheduler(scheduler)
        sigmas = self._custom_sigmas(scheduler, steps, denoise, width, height)
        if sigmas is None and scheduler not in comfy.samplers.KSampler.SCHEDULERS:
            scheduler = "normal"

        noise = self.make_noise(latent_image, seed, subseed, sub_strength,
                                seed_resize_from_h or 0, seed_resize_from_w or 0)
        logger.debug(f"SEED:{seed} STEPS:{steps} CFG:{cfg} SAMPLER:{sampler_name}/{scheduler} "
                     f"DENOISE:{denoise:.3f} SUBSEED:{subseed}@{sub_strength} {width}x{height}")
        samples = comfy.sample.sample(
            self.model, noise, steps, cfg, sampler_name, "simple" if sigmas is not None else scheduler,
            positive, negative, latent_image, denoise=denoise, sigmas=sigmas,
            # ancestral/SDE samplers seed their per-step noise from this; offset it (A1111's "eta noise
            # seed delta") or it equals the initial noise, which the NV RNG reproduces bit-for-bit
            seed=(seed + ETA_NOISE_SEED_DELTA) % (2 ** 63),
            last_step=last_step, disable_pbar=True,
        )
        decoded = self.decode_sample(self.vae, samples)
        image = Image.fromarray((decoded[0].float().cpu().numpy() * 255.0).round().astype(np.uint8)).convert("RGB")
        if return_latent:
            return samples, image
        return image

    # ---------------------------------------------------------------- cleanup
    def release_vram(self):
        """Move models off the GPU (ComfyUI reloads them on demand); keeps them in RAM."""
        import comfy.model_management
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        empty_cache()

    def cleanup(self):
        """Release the model, text encoders and VAE (they reload on the next frame)."""
        import comfy.model_management
        self.optimized = False
        self.model_loaded = False
        self.model = self.base_model = self.clip = self.base_clip = self.vae = None
        self.ip_adapter = self.clip_vision = None
        self._cond_cache.clear()
        self._pinned_conds = set()
        self.cond = self.n_cond = None
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        empty_cache()


def apply_controlnet(conditioning, control_net, image, strength):
    """Attach a ComfyUI ControlNet to a conditioning (image: [1, H, W, C] 0..1)."""
    if strength == 0:
        return conditioning
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
