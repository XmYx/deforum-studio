import shutil
import datetime
import os
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from loguru import logger as logurulogger
from deforum.utils.constants import config
import json

def save_test_configuration(test_path, configuration):
    config_path = os.path.join(test_path, 'configuration.json')
    with open(config_path, 'w') as config_file:
        json.dump(configuration, config_file, indent=4)
    logurulogger.info(f"Configuration saved: {config_path}")

def get_frames(video_path):
    """Extract frames from a given video path using OpenCV."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_psnr(frame1, frame2):
    """Calculate the PSNR between two frames."""
    mse = np.mean((frame1 - frame2) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compare_videos(path1, path2):
    logurulogger.info(f"Comparing videos: {path1} vs {path2}")
    frames1 = get_frames(path1)
    frames2 = get_frames(path2)

    if len(frames1) != len(frames2):
        logurulogger.error("Different number of frames")
        return False, "Different number of frames"

    ssim_scores = []
    psnr_scores = []
    for frame1, frame2 in zip(frames1, frames2):
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        score, _ = ssim(frame1_gray, frame2_gray, full=True)
        ssim_scores.append(score)

        # Calculate PSNR
        psnr_score = get_psnr(frame1_gray, frame2_gray)
        psnr_scores.append(psnr_score)

    average_ssim = sum(ssim_scores) / len(ssim_scores)
    average_psnr = sum(psnr_scores) / len(psnr_scores)


    logurulogger.info(f"Average SSIM: {average_ssim:.4f}")
    logurulogger.info(f"Average PSNR: {average_psnr:.4f}")

    # Define thresholds
    ssim_threshold = 0.75
    psnr_threshold = 30

    if average_ssim >= ssim_threshold and average_psnr >= psnr_threshold:
        logurulogger.info("Videos are similar based on set thresholds.")
        return True, f"Videos are similar. Avg SSIM: {average_ssim:.4f}, Avg PSNR: {average_psnr:.4f}"
    else:
        logurulogger.warning("Videos differ based on set thresholds.")
        return False, f"Videos differ. Avg SSIM: {average_ssim:.4f}, Avg PSNR: {average_psnr:.4f}"


def manage_test_storage(src_video_path, batch_name):
    """Manage storage of test videos and establish a baseline if not present."""
    tests_dir = f"{config.root_path}/tests"
    base_dir = f"{tests_dir}/base"
    current_test_dir = f"{tests_dir}/{datetime.datetime.now().strftime('%Y-%m-%d')}"

    # Ensure all directories exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(current_test_dir, exist_ok=True)

    base_video_path = f"{base_dir}/{batch_name}.mp4"
    test_video_path = f"{current_test_dir}/{batch_name}.mp4"

    # Establish or update the baseline as needed
    if not os.path.exists(base_video_path):
        shutil.copy(src_video_path, base_video_path)
        return base_video_path, test_video_path, False  # No comparison if baseline was just established
    else:
        # Copy video to current test directory
        shutil.copy(src_video_path, test_video_path)
    return base_video_path, test_video_path, True  # Proceed with comparison

def run_e2e_test(options, extra_args):
    """Run e2e tests using DeforumAnimationPipeline and manage test outputs."""
    from deforum import DeforumAnimationPipeline

    modelid = str(options.get("modelid", "125703"))
    deforum = DeforumAnimationPipeline.from_civitai(model_id=modelid)
    deforum.generator.optimize = True
    # Setup logging directory based on current date and time
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.datetime.now().strftime("%H-%M-%S")
    log_directory = f"{config.root_path}/logs/tests/{date_str}"
    os.makedirs(log_directory, exist_ok=True)
    log_file_path = f"{log_directory}/{time_str}.log"

    # Configure loguru logger
    logurulogger.add(log_file_path, rotation="10 MB", compression="zip", level="INFO")
    logurulogger.info("Logger configured and ready to record test results.")

    preset_dir = config.settings_path
    files = []
    for root, _, filenames in os.walk(preset_dir):
        for file in filenames:
            if file.endswith('.txt'):
                files.append(os.path.join(root, file))

    logurulogger.info(f"Running {len(files)} settings from path: {preset_dir}")
    logurulogger.debug(f"Full list of settings files to run: {' '.join(files)}")

    for file_path in files:
        try:           
            logurulogger.info(f"Settings file path: {file_path}")
            batch_name = file_path.split('.')[0].split("/")[-1]
            logurulogger.info(f"Batch Name: {batch_name}")

            extra_args["settings_file"] = file_path
            options.update({
                "prompts": {"0": "A solo delorean speeding on an ethereal highway through time jumps, like in the iconic movie back to the future."},
                "seed": 420,
                "subseed":1,
                "subseed_schedule":"0:(1)",
                "batch_name": batch_name,
                "dry_run": False,
                "max_frames": 10,
                "hybrid_use_full_video": False
            })
            start_time = time.time()
            animation = deforum(**extra_args, **options)
            elapsed_time = time.time() - start_time
            logurulogger.info(f"Elapsed time for {batch_name}: {elapsed_time:.2f} seconds")
            # Manage video storage and comparison
            base_video_path, test_video_path, do_compare = manage_test_storage(animation.video_path, batch_name)
            if do_compare:
                comparison_result = compare_videos(base_video_path, test_video_path)
                logurulogger.info(f"Comparison result for {batch_name}: {comparison_result}")
            else:
                logurulogger.info("Baseline established; no comparison needed.")
        except Exception as e:
            logurulogger.error(f"Exception during test run: {str(e)}")