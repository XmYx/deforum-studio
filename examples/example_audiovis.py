import contexttimer
import math
import numpy as np
import io
import os
import shutil
import subprocess
import threading
import time
import librosa
from subprocess import Popen
import mutagen

from deforum import DeforumAnimationPipeline
from deforum.pipeline_utils import load_settings
from deforum.utils.constants import config
from deforum.utils.logging_config import logger


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


def load_audio(audio_path):
    with open(audio_path, 'rb') as f:
        audio_data = io.BytesIO(f.read())
    return audio_data

def detect_beats(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    plp = librosa.beat.plp(y=y, sr=sr)
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    beats = np.maximum(beats - 0.25, 0)  # Adjust for prepended silence
    beat_strengths = plp[beat_frames]
    if beat_strengths.size > 0:
        beat_strengths = (beat_strengths - beat_strengths.min()) / (beat_strengths.max() - beat_strengths.min())
    return tempo, beats

def all(audio_path):
    audio_data = load_audio(audio_path)
    y, sr = librosa.load(audio_data, sr=None, mono=True)
    bmp, beats = detect_beats(y, sr)
    beat_offset = beats[0]
    return bmp, beat_offset


if __name__ == "__main__":

    start_time = time.time()

    ##### INPUT ARGS
    INPUT_AUDIO = "char_cut.mp3"
    BASE_DEFORUM_PRESET = 'atk-move-on-up.txt'
    OVERRIDE_FRAME_COUNT = 360
    WIDTH = 1024
    HEIGHT = 576
    PROMPT = {0: "cute slime mold in a forest 3d cats with lasers wearing cool daft punk armor and djing red eye creating beksiński rave herd surrounded dreamtime figure hardearned pill"}

    ##### LOAD ARGS
    args = load_settings(os.path.join(config.presets_path, 'settings', BASE_DEFORUM_PRESET))

    ##### MODIFY ARGS
    fps = args["fps"]
    milkdrop_preset = args["projectm_preset_name"]
    args["translation_y"] = "0: ((15 * sin((bpm / 30 * pi * (f + beat_offset_f) / fps))**1 + 15))",
    args["strength_schedule"] = "0: ((-0.17 * cos((bpm / 60 * pi * (f + beat_offset_f) / fps))**0.90 + 0.42))"
    args["subseed_strength_schedule"] = "0: ((0.1 * cos((bpm / 60 * pi * (f + beat_offset_f) / fps))**1 + 0.55))",



    # milkdrop_preset = args["projectm_preset_name"]

    ##### SETUP
    milkdrop_path = os.path.join(config.presets_path, 'projectm', milkdrop_preset)
    preset_dir = os.path.join(config.root_path,"presets")
    settings_dir = os.path.join(preset_dir,"settings")
    audio_file_path = INPUT_AUDIO
    job_name = f"manual_audiovis_{time.strftime('%Y%m%d%H%M%S')}"
    job_output_dir =  os.path.join(config.output_dir, job_name)
    hybrid_frame_path = os.path.join(job_output_dir, "inputframes")
    os.makedirs(hybrid_frame_path, exist_ok=True)

    ##### AUDIO PROCESSING
    events = detect_onsets(audio_file_path)
    bpm, beat_offset = all(audio_file_path)

    ##### MILKDROP PROCESSING
    projectm_process = run_projectm(
        input_audio = audio_file_path,
        host_output_path = hybrid_frame_path,
        preset = milkdrop_path,
        fps = fps
    )
    if projectm_process is None:
        logger.error("ProjectM process failed to start. Exiting.")
        exit(1)
    projectM_thread = threading.Thread(target=monitor_projectm, args=(projectm_process,))
    projectM_thread.start()

    ##### DEFORUM MODS
    expected_frame_count = OVERRIDE_FRAME_COUNT or math.floor(fps * get_audio_duration(audio_file_path))
    args["outdir"] = job_output_dir
    args["batch_name"] = job_name
    args["max_frames"] = expected_frame_count
    args["add_soundtrack"] = "File"
    args["soundtrack_path"] = audio_file_path
    args["schedule_events"] = events
    args["video_init_path"] = os.path.join(settings_dir, args["video_init_path"])
    args["bpm"] = bpm
    args["beat_offset"] = beat_offset
    args["width"] = WIDTH
    args["height"] = HEIGHT
    args["prompts"] = PROMPT

    ##### RUN DEFORUM
    pipeline = DeforumAnimationPipeline.from_civitai("125703")
    gen = pipeline(**args)
    logger.info(f"Output video: {gen.video_path}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
