"""
"World Model" animation mode: FLUX.3 Action predicts the animation's next frames.

Each step hands the current frame and the frame's prompt to the world model, which returns the next ~2 s
of video (32 frames at the model's 15 fps). Every predicted frame can then be refined with the regular image
model (img2img, ``world_model_refine_denoise``) so it keeps the prompt's look. The last frame of a chunk
seeds the next prediction, so the animation runs on its own predictions.

Frame 0 comes from, in order: the init image (``use_init``), the first frame of ``video_init_path``, or a
txt2img frame from the image model. With ``world_model_video_guided`` every chunk restarts from the init
video's frame at that point instead of the model's own last frame.
"""
import os

import cv2
import numpy as np
from PIL import Image

from deforum.utils.logging_config import logger

from ...pipeline_utils import next_seed
from ...utils.image_utils import load_image, save_image
from ...utils.string_utils import prepare_prompt

WORLD_MODEL_DEFAULTS = {
    "world_model_policy": "",            # Hub repo / dir; empty = WORLD_MODEL_POLICY or the DROID release
    "world_model_subfolder": None,       # package variant; None = WORLD_MODEL_SUBFOLDER or variants/gd-fp8r
    "world_model_refine_denoise": 0.35,  # 0 = raw world-model frames; >0 = img2img refine with the image model
    "world_model_video_guided": False,
    "world_model_guidance": None,        # None = the checkpoint's own guidance
    "world_model_steps": None,           # None = the checkpoint's own step count
}


def _setting(gen, key):
    value = getattr(gen, key, None)
    return WORLD_MODEL_DEFAULTS[key] if value in (None, "") else value


def _video_frame(path, index, size):
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(index, n - 1))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(size, Image.Resampling.LANCZOS)


def _emit(cls, image):
    """Store, save and stream one finished frame, then advance the frame counter (like post_gen)."""
    gen = cls.gen
    gen.image = image
    gen.opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gen.prev_img = gen.opencv_image
    cls.images.append(np.array(image))
    filename = f"{gen.timestring}_{gen.frame_idx:09}.png"
    save_image(image, "PIL", filename, gen, gen, gen, cls)
    gen.image_paths.append(os.path.join(gen.outdir, filename))
    cls.datacallback({"image": image, "operation_id": gen.operation_id, "frame_idx": gen.frame_idx,
                      "image_path": os.path.join(gen.outdir, filename), "is_interpolated": False})
    gen.frame_idx += 1
    gen.seed = next_seed(gen, gen)
    gen.image_count = len(cls.images)


def _first_frame(cls):
    gen = cls.gen
    size = (int(gen.width), int(gen.height))
    if gen.use_init and gen.init_image:
        img = gen.init_image if isinstance(gen.init_image, Image.Image) else load_image(gen.init_image)
        return img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    if getattr(gen, "video_init_path", "") and os.path.isfile(gen.video_init_path):
        frame = _video_frame(gen.video_init_path, 0, size)
        if frame is not None:
            return frame
    return cls.generate()  # txt2img with the image model


def world_model_step(cls):
    """Shoot function for animation_mode == 'World Model' (replaces warp + diffusion for this mode)."""
    gen = cls.gen
    if gen.frame_idx == 0 and not cls.images:
        _emit(cls, _first_frame(cls))
        return

    if getattr(cls, "world_model", None) is None:
        from deforum.worldmodel.client import WorldModelClient
        cls.world_model = WorldModelClient(policy=_setting(gen, "world_model_policy") or None,
                                           subfolder=_setting(gen, "world_model_subfolder"))
    size = (int(gen.width), int(gen.height))
    observed = Image.fromarray(cv2.cvtColor(gen.prev_img, cv2.COLOR_BGR2RGB))
    if _setting(gen, "world_model_video_guided") and os.path.isfile(getattr(gen, "video_init_path", "") or ""):
        observed = _video_frame(gen.video_init_path, gen.frame_idx, size) or observed

    prompt, _, _, _ = cls.frame_prompts(gen.frame_idx)
    if hasattr(cls.generator, "release_vram"):
        cls.generator.release_vram()  # the world model runs in another process and needs the VRAM now
    predicted = cls.world_model.predict(observed, prompt, seed=gen.seed,
                                        guidance=_setting(gen, "world_model_guidance"),
                                        steps=_setting(gen, "world_model_steps"))
    logger.info(f"World model predicted {len(predicted)} frames from frame {gen.frame_idx - 1}")

    denoise = float(_setting(gen, "world_model_refine_denoise"))
    for frame in predicted:
        if gen.frame_idx >= gen.max_frames or cls.interrupt:
            break
        frame = frame.resize(size, Image.Resampling.LANCZOS)
        if denoise > 0 and not gen.dry_run:
            # img2img through the pipeline's own generate(): same sampler mapping, styles, prompt
            # schedule and blending as the other modes, on this frame's prompt
            gen.prev_img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            gen.strength = 1.0 - denoise
            gen.prompt = prepare_prompt(str(gen.prompt_series[gen.frame_idx]), gen.max_frames, gen.seed,
                                        gen.frame_idx)
            frame = cls.generate()
        _emit(cls, frame)


def close_world_model(cls):
    wm = getattr(cls, "world_model", None)
    if wm is not None:
        wm.close()
        cls.world_model = None
