import os
import sys
import json
import math
import time
import subprocess
from pathlib import Path
from argparse import ArgumentParser
import imageio

from deforum import logger
from deforum.shared_storage import models
from deforum.utils.audio_utils.deforum_audio import get_audio_duration
from deforum.utils.file_utils.extract_nth_files import extract_nth_files
from deforum import DeforumAnimationPipeline

if 'deforum_pipe' not in models:
    models['deforum_pipe'] = DeforumAnimationPipeline.from_civitai(model_id="125703")

loaded_model_id = "125703"


def run_backend(settings_file):
    if not os.path.exists(settings_file):
        print(f"File '{settings_file}' does not exist.")
        return False

    params = {}

    # Load settings from file
    if settings_file.endswith(".txt"):
        try:
            with open(settings_file, "r") as f:
                params.update(json.load(f))
        except Exception as e:
            print(f"[ERROR] Failed to load settings file: {e}")
            return False
    else:
        print("Settings file must be a .txt file containing JSON.")
        return False

    # Defaults / overrides
    params["settings_file"] = settings_file
    models['deforum_pipe'].generator.optimize = params.get('optimize', True)

    # Parse prompts
    prom = params.get("prompts", "cat sushi")
    key = params.get("keyframes", "0")
    if not isinstance(prom, dict):
        prom_lines = prom.strip().split("\n")
        key_lines = key.strip().split("\n")
        params["animation_prompts"] = dict(zip(key_lines, prom_lines))
    else:
        params["animation_prompts"] = prom

    # Handle timestring
    timestring = time.strftime('%Y%m%d%H%M%S')
    params["timestring"] = timestring if not params.get("resume_from_timestring") else params["resume_timestring"]

    # Visualization pass (e.g., Milkdrop audio visuals)
    if params.get("generate_viz"):
        from deforum.utils.constants import config
        output_path = Path(config.root_path) / 'output' / 'deforum' / f"{params['batch_name']}_{timestring}" / 'inputframes'
        output_path.mkdir(parents=True, exist_ok=True)

        params['max_frames'] = int(math.floor(params['fps'] * get_audio_duration(params['audio_path'])) / params["extract_nth_frame"])

        command = (
            f"EGL_PLATFORM=surfaceless projectMCli "
            f"-a \"{params['audio_path']}\" "
            f"--presetFile \"{Path(config.root_path) / 'milks' / params['milk_path']}\" "
            f"--outputType image "
            f"--outputPath \"{output_path}/\" "
            f"--fps 24 --width {params['width']} --height {params['height']}"
        )
        subprocess.run(command, shell=True)

        if params["extract_nth_frame"] > 1:
            extract_nth_files(str(output_path), params["extract_nth_frame"])

        # Combine into video
        temp_video_path = Path(config.root_path) / 'temp_video.mp4'
        final_output_path = Path(config.root_path) / 'output.mp4'
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

        image_files = sorted(Path(output_path).glob("*.jpg"), key=lambda x: int(x.stem))
        writer = imageio.get_writer(str(temp_video_path), fps=24)
        for image_path in image_files:
            writer.append_data(imageio.imread(image_path))
        writer.close()

        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", str(temp_video_path),
            "-i", params["audio_path"],
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            "-shortest",
            str(final_output_path)
        ]
        subprocess.run(ffmpeg_command, text=True)
        print(f"Generated video at: {final_output_path}")
        return True

    # Advanced diffusion pass
    if params.get("enable_ad_pass"):
        params["adiff_pass_params"] = {
            "max_frames": params['ad_max_frames'],
            "use_every_nth": params['ad_use_every_nth'],
            "width": params['width'],
            "height": params['height'],
            "closed_loop": params['ad_closed_loop'],
            "prompt": params['ad_prompt'],
            "negative_prompt": params['ad_negative_prompt'],
            "hires_steps": params['ad_hires_steps'],
            "hires_pass_denoise": 0.56,
            "controlnet_strength": params["ad_controlnet_strength"],
            "controlnet_start": 0.0,
            "controlnet_end": 0.5,
            "ip_adapter_video_strength": params["ad_ip_adapter_video_strength"],
            "ip_adapter_video_start": 0.0,
            "ip_adapter_video_end": 0.8,
            "ip_adapter_image_strength": params["ad_ip_adapter_image_strength"],
            "ip_adapter_image_start": 0.0,
            "ip_adapter_image_end": 0.5,
            "seed": params["ad_seed"],
            "steps": params["ad_steps"],
            "cfg": params["ad_cfg"],
            "sampler": params["ad_sampler_name"],
            "scheduler": params["ad_scheduler"],
            "start_at_step": params["ad_start_step"],
            "fps": params["fps"],
            "sd_model": params["ad_sd_model"],
            "lora": params["ad_lora"],
            "ad_model": params["ad_model"],
            "sampler_name": params["ad_sampler_name"],
            "beta_schedule": params["ad_beta_schedule"],
        }

    # Generate animation
    animation = models['deforum_pipe'](callback=None, **params)

    result = {
        "status": "Ready",
        "timestring": animation.timestring,
        "resume_path": animation.outdir,
        "resume_from": animation.max_frames
    }

    if hasattr(animation, 'video_path'):
        result["video_path"] = animation.video_path

    print("Animation generation complete:")
    print(json.dumps(result, indent=2))
    return True


def main():
    parser = ArgumentParser(description="Run Deforum pipeline with a settings file.")
    parser.add_argument("settings_file", type=str, help="Path to the .txt settings file")
    args = parser.parse_args()
    success = run_backend(args.settings_file)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
