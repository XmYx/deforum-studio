"""
Model discovery and pipeline construction shared by the Qt UI, the headless runner and the CLI.

Models are looked up in ComfyUI's folders (see generators/comfy_utils.py for the search paths):
full checkpoints (``checkpoints``) and diffusion-model-only releases (``diffusion_models``) are
both selectable; text encoders and VAEs of split releases are matched automatically.
"""
import os

from deforum.utils.logging_config import logger

AUTO = "Auto"
CIVITAI = "CivitAI model id"
_MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".gguf")


def _names(kind):
    from deforum.generators.comfy_utils import MODEL_KINDS, ensure_comfy
    ensure_comfy()
    import folder_paths
    # files of other kinds seen through the flat ~/deforum/models checkpoint folder
    foreign = {sub + "/" for k, subs in MODEL_KINDS.items() for sub in subs if sub} | {"other/"}
    try:
        names = folder_paths.get_filename_list(kind)
    except KeyError:
        return []
    return sorted(n for n in names if n.lower().endswith(_MODEL_EXTS)
                  and not any(n.replace("\\", "/").startswith(f) for f in foreign))


def list_models():
    """Selectable image models: full checkpoints, then diffusion-model-only files."""
    checkpoints = _names("checkpoints")
    return checkpoints + [n for n in _names("diffusion_models") if n not in checkpoints]


def list_vaes():
    return [AUTO] + _names("vae")


def list_samplers():
    from deforum.generators.comfy_utils import ensure_comfy
    ensure_comfy()
    import comfy.samplers
    return list(comfy.samplers.KSampler.SAMPLERS)


def list_schedulers():
    from deforum.generators.comfy_utils import ensure_comfy
    ensure_comfy()
    import comfy.samplers
    return list(comfy.samplers.KSampler.SCHEDULERS) + ["flux2"]


def default_model():
    """Prefer an SDXL-class checkpoint, else whatever comes first."""
    models = list_models()
    for hint in ("xl", "flux", "sd"):
        for m in models:
            if hint in m.lower():
                return m
    return models[0] if models else None


def resolve_model(name):
    """Model name from the UI / settings -> absolute path (None if unknown)."""
    if not name or name in (AUTO, CIVITAI):
        return None
    if os.path.isfile(name):
        return name
    from deforum.generators.comfy_utils import ensure_comfy
    ensure_comfy()
    import folder_paths
    return folder_paths.get_full_path("checkpoints", name) or folder_paths.get_full_path("diffusion_models", name)


def get_pipeline(params, models_store):
    """Return the shared DeforumAnimationPipeline, pointing its generator at the selected model.

    ``params`` keys used: ``model`` (file name or path), ``vae`` (name or "Auto"), ``text_encoders``
    (optional list), ``clip_type`` (optional), ``model_id`` (CivitAI id when model == CIVITAI).
    Loading is lazy: weights load on the first frame and stay in RAM between runs.
    """
    from deforum import DeforumAnimationPipeline
    from deforum.generators.comfy_sd_generator import ComfyDeforumGenerator

    model = params.get("model") or ""
    if model == CIVITAI and params.get("model_id"):
        from deforum.utils.constants import config
        from deforum.utils.model_download import download_from_civitai
        model_path = os.path.join(config.model_dir, download_from_civitai(model_id=str(params["model_id"]),
                                                                          destination=config.model_dir))
    else:
        model_path = resolve_model(model) or resolve_model(default_model())
    if model_path is None:
        raise RuntimeError("No model found. Put a checkpoint into ~/deforum/models/checkpoints or set "
                           "EXTRA_MODEL_PATHS to a ComfyUI models folder.")
    vae = params.get("vae")
    vae = None if not vae or vae == AUTO else vae

    pipe = models_store.get("deforum_pipe")
    if pipe is None:
        pipe = DeforumAnimationPipeline(generator=ComfyDeforumGenerator(
            model_path=model_path, vae_path=vae, text_encoders=params.get("text_encoders"),
            clip_type=params.get("clip_type")))
        models_store["deforum_pipe"] = pipe
    elif pipe.generator.set_components(model_path=model_path, vae_path=vae or "",
                                       text_encoders=params.get("text_encoders") or [],
                                       clip_type=params.get("clip_type") or ""):
        logger.info(f"Switched model to {os.path.basename(model_path)}")
    return pipe
