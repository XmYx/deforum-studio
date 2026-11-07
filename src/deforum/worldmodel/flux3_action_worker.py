"""
FLUX.3 Action world-model worker.

Runs in its own Python 3.12 environment with `flux-action` installed (it pins torch / transformers / numpy
versions that differ from Deforum's), started and driven by ``deforum.worldmodel.client``. It never imports
Deforum.

FLUX.3 Action denoises the next chunk of actions *together with* the next video frames, from the current
camera frame, a state vector and a text instruction. Its inference API returns only the actions; this worker
runs the same sampler and keeps the predicted video latents, decodes them with the model's own video VAE and
returns them as frames. That makes the policy usable as a world model: frame in, the next ~2 s of frames out.

Protocol: one JSON object per line on stdin, one JSON reply per line on stdout (library output goes to
stderr). Frames are exchanged as PNG files.

  {"cmd": "load", "policy": "<hub repo or dir>", "subfolder": null, "offload_text_encoder": true,
   "compile": false}
  {"cmd": "predict", "image": "<png>", "caption": "...", "seed": 0, "out_dir": "<dir>",
   "guidance": null, "steps": null}            -> {"ok": true, "frames": ["<png>", ...], "fps": 15.0}
  {"cmd": "info"} / {"cmd": "quit"}
"""
import json
import os
import sys
import traceback

PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr  # keep library prints off the protocol channel

import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from PIL import Image  # noqa: E402

POLICY = None


def reply(**kw):
    PROTOCOL_OUT.write(json.dumps(kw) + "\n")
    PROTOCOL_OUT.flush()


def load(policy, subfolder=None, revision=None, offload_text_encoder=True, compile=False, device=None):
    global POLICY
    from flux_action.policy import FluxActionPolicy

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Load on the CPU and park the 8.9 GB text encoder there before moving to the GPU, so the DiT + VAE fit
    # next to the image model; the encoder visits the GPU only to encode a caption it has not seen yet.
    POLICY = FluxActionPolicy.from_pretrained(policy, device="cpu", subfolder=subfolder, revision=revision)
    if offload_text_encoder:
        POLICY.set_text_encoder_offload(True)
    POLICY.to(device)
    prepared = POLICY.prepare_inference(compile=compile)
    cfg = POLICY.config
    return {"device": str(POLICY.device), "prepared": bool(prepared), "layout": cfg.camera_layout,
            "canvas_hw": list(cfg.canvas_hw), "latent_hw": list(cfg.latent_hw), "fps": cfg.fps,
            "chunk_size": cfg.chunk_size, "steps": cfg.num_inference_steps}


def _canvas(frame):
    """RGB uint8 (H, W, 3) -> canvas (3, 1, Hc, Wc) in [-1, 1] for the policy's camera layout."""
    from flux_action.processing import packing

    cfg = POLICY.config
    x = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)[None]  # (1, 3, H, W)
    if cfg.camera_layout == "droid":
        # the frame fills the whole 540x640 composite (where the wrist + exterior views would be)
        comp = torch.nn.functional.interpolate(x, size=packing.COMPOSITE_HW, mode="bilinear",
                                               align_corners=False, antialias=True)
        return packing.pad_composite(comp, cfg.canvas_hw)
    # single / side_by_side / grid: one camera, repeated if the layout expects several
    n_cams = {"single": 1, "side_by_side": 2}.get(cfg.camera_layout, 1)
    cams = x[None].repeat(n_cams, 1, 1, 1, 1)  # (n_cams, T=1, 3, H, W)
    return packing.compose_canvas(cams, cfg.camera_layout, cfg.canvas_hw)


@torch.no_grad()
def predict(image, caption, seed, out_dir, guidance=None, steps=None):
    from flux_action.processing import packing

    cfg = POLICY.config
    if guidance is not None:
        cfg.guidance_scale = float(guidance)
    if steps is not None:
        cfg.num_inference_steps = int(steps)
    src = Image.open(image).convert("RGB")
    frame = np.array(src)  # writable copy for torch.from_numpy
    state = torch.zeros(1, cfg.action_dim, device=POLICY.device)
    cond = POLICY._encode_canvas(_canvas(frame), POLICY._normalize_state(state))

    # Run the policy's own joint video+action sampler and keep the video it would otherwise discard. Both
    # the prepared (CUDA) and the reference path end in one of these solvers, so wrap them for this call.
    from flux_action.inference import sampling

    captured = {}
    originals = {name: getattr(sampling, name) for name in ("euler", "cosmos_unipc_order2")}

    def capturing(fn):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            captured["x_video"] = out["x_video"]
            return out
        return wrapped

    for name, fn in originals.items():
        setattr(sampling, name, capturing(fn))
    try:
        POLICY._sample(cond, caption, int(seed) % (2 ** 63))
    finally:
        for name, fn in originals.items():
            setattr(sampling, name, fn)
    if "x_video" not in captured:
        raise RuntimeError("the policy's sampler returned no video stream")

    h, w = cfg.latent_hw
    pred = rearrange(captured["x_video"].float(), "b (t h w) c -> b c t h w", h=h, w=w)
    observed = rearrange(cond["x_video_cond"].float(), "b (t h w) c -> b c t h w", h=h, w=w)
    latents = torch.cat([observed, pred], dim=2).to(torch.bfloat16)
    video = POLICY.video_vae.decode(latents).float()  # (1, 3, T, H, W) in [-1, 1]
    ch, cw = packing.content_hw(cfg.camera_layout, cfg.canvas_hw)
    video = video[0, :, 1:, :ch, :cw].clamp(-1, 1).add(1).mul(127.5)  # drop the observed frame itself
    frames = video.permute(1, 2, 3, 0).round().byte().cpu().numpy()  # (T, H, W, 3)

    del video, latents, pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # hand the decode buffers back to the image model in the other process
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, f in enumerate(frames):
        p = os.path.join(out_dir, f"wm_{i:04d}.png")
        Image.fromarray(f).resize(src.size, Image.Resampling.LANCZOS).save(p, compress_level=1)
        paths.append(p)
    return paths


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            cmd = req.pop("cmd")
            if cmd == "load":
                reply(ok=True, info=load(**req))
            elif cmd == "predict":
                if POLICY is None:
                    raise RuntimeError("no policy loaded")
                reply(ok=True, frames=predict(**req), fps=POLICY.config.fps)
            elif cmd == "info":
                reply(ok=True, loaded=POLICY is not None, torch=torch.__version__,
                      cuda=torch.cuda.is_available())
            elif cmd == "quit":
                reply(ok=True)
                return
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except Exception as e:  # report the error and keep serving
            traceback.print_exc()
            reply(ok=False, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
