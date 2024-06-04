import contexttimer
import math
import numpy as np
import os
import shutil
import subprocess
import tempfile
import threading
import time
import librosa
from subprocess import Popen

import mutagen
import requests

from deforum import DeforumAnimationPipeline
from deforum.pipeline_utils import load_settings
from deforum.utils.constants import config
from deforum.utils.logging_config import logger


#############
# Setup
INPUT_AUDIO = "https://vizrecord.app/audio/ArtThing.mp3"
MILKDROP_PRESET = os.path.join(config.presets_path, 'projectm', 'rewbs-04.milk')
BASE_DEFORUM_PRESET = os.path.join(config.presets_path, 'settings', 'Classic-Zoom-In.txt')
FPS = 20
WIDTH = 1024
HEIGHT = 576
OVERRIDE_FRAME_COUNT = 640 # limit frame count for testing, set to None to generate full length
#############

def run_projectm(input_audio: str, host_output_path : str, preset : str, fps: int = 20, width: int = 1024, height: int = 576) -> Popen[bytes]:

    logger.info(f"Starting projectM. Writing frames to: {host_output_path}")

    assert os.path.exists(input_audio) and os.path.isfile(input_audio)
    assert os.path.exists(host_output_path) and os.path.isdir(host_output_path)

    # Update with your path to 'texture' subdirectory of https://github.com/projectM-visualizer/presets-milkdrop-texture-pack
    texture_path = "/home/rewbs/milkdrop/textures"
    projectm_path = config.projectm_executable

    if not shutil.which(projectm_path):
        logger.error("No projectm executable found. Tried: " + projectm_path)
        return
    if not os.path.exists(preset):
        logger.error("No projectM preset found at: " + preset)
        return
    if not os.path.exists(texture_path):
        # Not fatal but may affect output of some presets that depend on textures.
        logger.warning("No projectM texture directory found. Some presets may not render as expected. Tried: " + texture_path)

    command = [
        projectm_path,
        "--outputPath", f"{host_output_path}",
        "--outputType", "both",
        "--texturePath", f"{texture_path}",
        "--width", f"{width}",
        "--height", f"{height}",
        "--beatSensitivity", "2.0",
        "--calibrate", "1",
        "--fps", f"{fps}",
        "--presetFile",  preset,
        "--audioPath", f"{input_audio}"
    ]

    # Start the process (without blocking)
    logger.info("Running projectM with command: " + " ".join(command))
    projectm_env = os.environ.copy()
    projectm_env["EGL_PLATFORM"] = "surfaceless"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=projectm_env)

    return process

def monitor_projectm(process : Popen[bytes]):
    while True:
        logger.info("ProjectM running...")
        output = process.stdout.readline()
        error = process.stderr.readline()
        if output:
            logger.info("[PROJECTM - stdout] - " +  output.decode().strip())
        if error:
            logger.error("[PROJECTM - stderr] - " +  error.decode().strip())

        if process.poll() is not None:
            output = process.stdout.read()
            error = process.stderr.read()
            if output:
                logger.info("[PROJECTM - stdout] - " +  output.decode().strip())
            if error:
                logger.error("[PROJECTM - stderr] - " +  error.decode().strip())
            if (process.returncode != 0):
                logger.error(f"ProjectM exited with code {process.returncode}")
            else:
                logger.info("ProjectM completed successfully")
            break

        time.sleep(1)


def get_audio_duration(audio_file):
    audio = mutagen.File(audio_file)
    return audio.info.length


def detect_onsets(audio_file_path):
    with contexttimer.Timer() as onset_timer:
        y, sr = librosa.load(audio_file_path, sr=None, mono=True)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onsets = librosa.frames_to_time(onset_frames, sr=sr)
        onset_strengths = onset_env[onset_frames]
        if onset_strengths.size > 0:
            onset_strengths = (onset_strengths - onset_strengths.min()) / (onset_strengths.max() - onset_strengths.min())
        events = []
        for onset in onsets.tolist():
            events.append({
                "time": onset,
                
            })

    logger.info(f'Onset detection in: {onset_timer.elapsed}s')
    return events


if __name__ == "__main__":

    audio_file_path = None
    if INPUT_AUDIO.startswith("http"):
        requests.get(INPUT_AUDIO)
        audio_file_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        with open(audio_file_path, "wb") as file:
            file.write(requests.get(INPUT_AUDIO).content)
    else:
        audio_file_path = INPUT_AUDIO

    expected_frame_count = OVERRIDE_FRAME_COUNT or math.floor(FPS * get_audio_duration(audio_file_path))

    events = detect_onsets(audio_file_path)
    logger.info(events)


    job_name = f"manual_audiovis_{time.strftime('%Y%m%d%H%M%S')}"
    job_output_dir =  os.path.join(config.output_dir, job_name)
    hybrid_frame_path = os.path.join(job_output_dir, "inputframes")
    os.makedirs(hybrid_frame_path, exist_ok=True)

    # Start projectM and monitor it on a background thread.
    projectm_process = run_projectm(
        input_audio = audio_file_path,
        host_output_path = hybrid_frame_path,
        preset = MILKDROP_PRESET,
        fps = FPS
    )
    if projectm_process is None:
        logger.error("ProjectM process failed to start. Exiting.")
        exit(1)
    projectM_thread = threading.Thread(target=monitor_projectm, args=(projectm_process,))
    projectM_thread.start()

    # Run Deforum pipeline on main thread while projectM runs in the background
    pipeline = DeforumAnimationPipeline.from_civitai("125703")

    args = load_settings(BASE_DEFORUM_PRESET)
    args["outdir"] = job_output_dir
    args["batch_name"] = job_name
    args["max_frames"] = expected_frame_count
    args["width"] = WIDTH
    args["height"] = HEIGHT
    args["fps"] = FPS
    args["add_soundtrack"] = "File"
    args["soundtrack_path"] = audio_file_path
    args["schedule_events"] = events
    #args["dry_run"] = True
    args["translation_z"] = "(0): 6*(1-progress_until_next_event)"
    args["rotation_3d_y"] = "(0): (where(events_passed%2==0, 1, -1) * (1-progress_until_next_event) * 3)"
    args["translation_x"] = "(0): (where(events_passed%2==0, -1, 1) * (1-progress_until_next_event) * 3 * (1024/90))"
    # args["seed"] = 10
    args["sampler"] ="DPM++ SDE Karras"
    args["prompts"] = {"0": "Mind-blowing splish splosh splash of metallic paint, irridescent, viscous, satisfying, cinematic lighting, studio professional macrophotography."}
    args["hybrid_generate_inputframes"] = False
    args["hybrid_composite"] = "Normal"
    args["hybrid_comp_alpha_schedule"] = "0:(0.2)"
    args["hybrid_motion"] = "Optical Flow"
    args["hybrid_flow_factor_schedule"] = "0:(1)"
    #args["hybrid_motion_use_prev_img"] = True
    args["hybrid_use_first_frame_as_init_image"] = False
    args["hybrid_flow_method"] = "Farneback"

    gen = pipeline(**args)

    logger.info(f"Output video: {gen.video_path}")
