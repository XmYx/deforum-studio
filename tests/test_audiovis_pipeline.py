from deforum import DeforumAnimationPipeline
import subprocess
import threading
import os
import time
import math
import shutil
from subprocess import Popen
import mutagen
from deforum.utils.constants import config
from deforum.utils.logging_config import logger

def run_projectm(input_audio: str, host_output_path : str, preset: str = 'ORB - Chop chop.milk', fps: int = 20, width: int = 1024, height: int = 576) -> Popen[bytes]:

    logger.info(f"Starting projectM. Writing frames to: {host_output_path}")

    assert os.path.exists(input_audio) and os.path.isfile(input_audio)
    assert os.path.exists(host_output_path) and os.path.isdir(host_output_path)

    # Update with your path to 'texture' subdirectory of https://github.com/projectM-visualizer/presets-milkdrop-texture-pack
    texture_path = "/home/rewbs/milkdrop/textures"
    # Update with your path to a directory holding all milkdrop presets of interest
    preset_path = "/home/rewbs/milkdrop/presets_all"
    # Update with your path to the projectM binary
    projectm_path = config.projectm_executable
    

    if not os.path.exists(texture_path):
        logger.warning("No projectM texture directory found. Some presets may not render as expected. Tried: " + preset_path)
    if not os.path.exists(preset_path):
        logger.error("No projectM preset directory found. Tried: " + preset_path)
        return    
    if not shutil.which(projectm_path):
        logger.error("No projectm executable found. Tried: " + projectm_path)
        return

    command = [
        projectm_path,
        "--outputPath", f"{host_output_path}",
        "--outputType", "image",
        "--texturePath", f"{texture_path}",
        "--width", f"{width}",
        "--height", f"{height}",
        "--beatSensitivity", "2.0",
        "--calibrate", "1",
        "--fps", f"{fps}",
        "--presetFile",  os.path.join(preset_path, preset),
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

if __name__ == "__main__":
    
    #############
    # User settings
    preset = 'Flexi, martin + geiss - dedicated to the sherwin maxawow.milk'
    input_audio = "/home/rewbs/120bpm.mp3"
    fps = 24
    width = 1024
    height = 576
    #############

    expected_frame_count = math.floor(fps * get_audio_duration(input_audio))
    expected_frame_count = 24 # override max fames for testing

    job_name = f"manual_audiovis_{time.strftime('%Y%m%d%H%M%S')}"
    job_output_dir =  os.path.join(config.output_dir, job_name)
    hybrid_frame_path = os.path.join(job_output_dir, "inputframes")
    os.makedirs(hybrid_frame_path, exist_ok=True)   

    # Start projectM and monitor it on a background thread.
    projectm_process = run_projectm(
        input_audio = input_audio,
        host_output_path = hybrid_frame_path,
        preset = preset,
        fps = fps
    )
    if projectm_process is None:
        logger.error("ProjectM process failed to start. Exiting.")
        exit(1)
    thread = threading.Thread(target=monitor_projectm, args=(projectm_process,))
    thread.start()

    # Run Deforum pipeline on main thread while projectM runs in the background
    pipeline = DeforumAnimationPipeline.from_civitai("125703")

    args = {
        "outdir": job_output_dir,
        "width": width,
        "height": height,
        "show_info_on_ui": True,
        "tiling": False,
        "restore_faces": False,
        "seed_resize_from_w": 0,
        "seed_resize_from_h": 0,
        "seed": 10,
        "sampler": "DPM++ SDE Karras",
        "steps": 18,
        "batch_name": job_name,
        "seed_behavior": "fixed",
        "seed_iter_N": 1,
        "use_init": False,
        "strength": 0.8,
        "strength_0_no_init": True,
        "init_image": "https://deforum.github.io/a1/I1.png",
        "use_mask": False,
        "use_alpha_as_mask": False,
        "mask_file": "https://deforum.github.io/a1/M1.jpg",
        "invert_mask": False,
        "mask_contrast_adjust": 1.0,
        "mask_brightness_adjust": 1.0,
        "overlay_mask": True,
        "mask_overlay_blur": 4,
        "fill": 1,
        "full_res_mask": True,
        "full_res_mask_padding": 4,
        "reroll_blank_frames": "ignore",
        "reroll_patience": 10.0,
        "motion_preview_mode": False,
        "prompts": {
            "0": "eyeballs, mushrooms, dramatic framing, cinematic lighting, high quality professional photography, macrophotography, iris, corona, haze, ultra detailed, beautiful textures"
        },
        "animation_prompts_positive": "",
        "animation_prompts_negative": "skull, nsfw, nude, ugly, blurry, boring, signature, logo, writing, face, woman, girl, child, person, man",
        "animation_mode": "2D",
        "max_frames": expected_frame_count,
        "border": "replicate",
        "angle": "0: (0)",
        "zoom": "0:(1)",
        "translation_x": "0: (0)",
        "translation_y": "0: (0)",
        "translation_z": "0: (1.75)",
        "transform_center_x": "0: (0.5)",
        "transform_center_y": "0: (0.5)",
        "rotation_3d_x": "0: (0)",
        "rotation_3d_y": "0: (0)",
        "rotation_3d_z": "0: (0)",
        "enable_perspective_flip": False,
        "perspective_flip_theta": "0: (0)",
        "perspective_flip_phi": "0: (0)",
        "perspective_flip_gamma": "0: (0)",
        "perspective_flip_fv": "0: (53)",
        "noise_schedule": "0: (0)",
        "strength_schedule": "0: (0.55)",
        "contrast_schedule": "0: (1.0)",
        "cfg_scale_schedule": "0: (7)",
        "enable_steps_scheduling": False,
        "steps_schedule": "0: (25)",
        "fov_schedule": "0: (70)",
        "aspect_ratio_schedule": "0: (1)",
        "aspect_ratio_use_old_formula": False,
        "near_schedule": "0: (200)",
        "far_schedule": "0: (10000)",
        "seed_schedule": "0:(s), 1:(-1), \"max_f-2\":(-1), \"max_f-1\":(s)",
        "pix2pix_img_cfg_scale_schedule": "0:(1.5)",
        "enable_subseed_scheduling": True,
        "subseed_schedule": "0: (2)",
        "subseed_strength_schedule": "0: ((0.350 * sin((120 / 240 * 3.141 * (t + 0) / 24))**1 + 0.55))",
        "enable_sampler_scheduling": False,
        "sampler_schedule": "0: (\"Euler a\")",
        "use_noise_mask": False,
        "mask_schedule": "0: (\"{video_mask}\")",
        "noise_mask_schedule": "0: (\"{video_mask}\")",
        "enable_checkpoint_scheduling": False,
        "checkpoint_schedule": "0: (\"model1.ckpt\"), 100: (\"model2.safetensors\")",
        "enable_clipskip_scheduling": False,
        "clipskip_schedule": "0: (2)",
        "enable_noise_multiplier_scheduling": True,
        "noise_multiplier_schedule": "0: (1)",
        "resume_from_timestring": False,
        "resume_timestring": "",
        "enable_ddim_eta_scheduling": False,
        "ddim_eta_schedule": "0: (0)",
        "enable_ancestral_eta_scheduling": False,
        "ancestral_eta_schedule": "0: (1)",
        "amount_schedule": "0: (0.1)",
        "kernel_schedule": "0: (5)",
        "sigma_schedule": "0: (1)",
        "threshold_schedule": "0: (0)",
        "color_coherence": "None",
        "color_coherence_image_path": "",
        "color_coherence_video_every_N_frames": 1,
        "color_force_grayscale": False,
        "legacy_colormatch": False,
        "diffusion_cadence": 1,
        "optical_flow_cadence": "None",
        "cadence_flow_factor_schedule": "0: (1)",
        "optical_flow_redo_generation": "None",
        "redo_flow_factor_schedule": "0: (1)",
        "diffusion_redo": "0",
        "noise_type": "perlin",
        "perlin_octaves": 4,
        "perlin_persistence": 0.5,
        "use_depth_warping": True,
        "depth_algorithm": "Midas-3-Hybrid",
        "midas_weight": 0.2,
        "padding_mode": "border",
        "sampling_mode": "bicubic",
        "save_depth_maps": False,
        "video_init_path": "",
        "extract_nth_frame": 1,
        "extract_from_frame": 0,
        "extract_to_frame": -1,
        "overwrite_extracted_frames": False,
        "use_mask_video": False,
        "video_mask_path": "https://deforum.github.io/a1/VM1.mp4",
        "hybrid_comp_alpha_schedule": "0:(0.8)",
        "hybrid_comp_mask_blend_alpha_schedule": "0:(0.5)",
        "hybrid_comp_mask_contrast_schedule": "0:(1)",
        "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": "0:(100)",
        "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": "0:(0)",
        "hybrid_flow_factor_schedule": "0:(1)",
        "hybrid_generate_inputframes": False,
        "hybrid_generate_human_masks": "None",
        "hybrid_use_first_frame_as_init_image": True,
        "hybrid_motion": "Optical Flow",
        "hybrid_motion_use_prev_img": True,
        "hybrid_flow_consistency": True,
        "hybrid_consistency_blur": 16,
        "hybrid_flow_method": "Farneback",
        "hybrid_composite": "Normal",
        "hybrid_use_init_image": False,
        "hybrid_comp_mask_type": "None",
        "hybrid_comp_mask_inverse": False,
        "hybrid_comp_mask_equalize": "None",
        "hybrid_comp_mask_auto_contrast": False,
        "hybrid_comp_save_extra_frames": False,
        "parseq_manifest": "",
        "parseq_use_deltas": True,
        "parseq_non_schedule_overrides": False,
        "use_looper": False,
        "init_images": "{\n    \"0\": \"https://deforum.github.io/a1/Gi1.png\",\n    \"max_f/4-5\": \"https://deforum.github.io/a1/Gi2.png\",\n    \"max_f/2-10\": \"https://deforum.github.io/a1/Gi3.png\",\n    \"3*max_f/4-15\": \"https://deforum.github.io/a1/Gi4.jpg\",\n    \"max_f-20\": \"https://deforum.github.io/a1/Gi1.png\"\n}",
        "image_strength_schedule": "0:(0.75)",
        "blendFactorMax": "0:(0.35)",
        "blendFactorSlope": "0:(0.25)",
        "tweening_frames_schedule": "0:(20)",
        "color_correction_factor": "0:(0.075)",
        "cn_1_overwrite_frames": True,
        "cn_1_vid_path": "/home/rewbs/Lazerpart.mp4",
        "cn_1_mask_vid_path": "",
        "cn_1_enabled": False,
        "cn_1_low_vram": False,
        "cn_1_pixel_perfect": True,
        "cn_1_module": "canny",
        "cn_1_model": "diffusers_xl_canny_full [2b69fca4]",
        "cn_1_weight": "0:(0.6)",
        "cn_1_guidance_start": "0:(0.0)",
        "cn_1_guidance_end": "0:(1.0)",
        "cn_1_processor_res": 512,
        "cn_1_threshold_a": 100,
        "cn_1_threshold_b": 200,
        "cn_1_resize_mode": "Inner Fit (Scale to Fit)",
        "cn_1_control_mode": "ControlNet is more important",
        "cn_1_loopback_mode": False,
        "cn_2_overwrite_frames": True,
        "cn_2_vid_path": "",
        "cn_2_mask_vid_path": "",
        "cn_2_enabled": False,
        "cn_2_low_vram": False,
        "cn_2_pixel_perfect": True,
        "cn_2_module": "CLIP-ViT-H (IPAdapter)",
        "cn_2_model": "ip-adapter-plus_sdxl_vit-h [bc449f62]",
        "cn_2_weight": "0:(0.6)",
        "cn_2_guidance_start": "0:(0.0)",
        "cn_2_guidance_end": "0:(1)",
        "cn_2_processor_res": 0.5,
        "cn_2_threshold_a": 0.5,
        "cn_2_threshold_b": 0.5,
        "cn_2_resize_mode": "Inner Fit (Scale to Fit)",
        "cn_2_control_mode": "ControlNet is more important",
        "cn_2_loopback_mode": True,
        "cn_3_overwrite_frames": True,
        "cn_3_vid_path": "/home/rewbs/Lazerpart.mp4",
        "cn_3_mask_vid_path": "",
        "cn_3_enabled": False,
        "cn_3_low_vram": False,
        "cn_3_pixel_perfect": True,
        "cn_3_module": "depth_zoe",
        "cn_3_model": "sai_xl_depth_256lora [73ad23d1]",
        "cn_3_weight": "0:(1)",
        "cn_3_guidance_start": "0:(0.0)",
        "cn_3_guidance_end": "0:(1.0)",
        "cn_3_processor_res": 512,
        "cn_3_threshold_a": 0.5,
        "cn_3_threshold_b": 0.5,
        "cn_3_resize_mode": "Inner Fit (Scale to Fit)",
        "cn_3_control_mode": "ControlNet is more important",
        "cn_3_loopback_mode": False,
        "cn_4_overwrite_frames": True,
        "cn_4_vid_path": "",
        "cn_4_mask_vid_path": "",
        "cn_4_enabled": False,
        "cn_4_low_vram": False,
        "cn_4_pixel_perfect": False,
        "cn_4_module": "none",
        "cn_4_model": "None",
        "cn_4_weight": "0:(1)",
        "cn_4_guidance_start": "0:(0.0)",
        "cn_4_guidance_end": "0:(1.0)",
        "cn_4_processor_res": 64,
        "cn_4_threshold_a": 64,
        "cn_4_threshold_b": 64,
        "cn_4_resize_mode": "Inner Fit (Scale to Fit)",
        "cn_4_control_mode": "Balanced",
        "cn_4_loopback_mode": False,
        "cn_5_overwrite_frames": True,
        "cn_5_vid_path": "",
        "cn_5_mask_vid_path": "",
        "cn_5_enabled": False,
        "cn_5_low_vram": False,
        "cn_5_pixel_perfect": False,
        "cn_5_module": "none",
        "cn_5_model": "None",
        "cn_5_weight": "0:(1)",
        "cn_5_guidance_start": "0:(0.0)",
        "cn_5_guidance_end": "0:(1.0)",
        "cn_5_processor_res": 64,
        "cn_5_threshold_a": 64,
        "cn_5_threshold_b": 64,
        "cn_5_resize_mode": "Inner Fit (Scale to Fit)",
        "cn_5_control_mode": "Balanced",
        "cn_5_loopback_mode": False,
        "freeu_enabled": True,
        "freeu_b1": "0:(1.3)",
        "freeu_b2": "0:(1.4)",
        "freeu_s1": "0:(0.9)",
        "freeu_s2": "0:(0.2)",
        "kohya_hrfix_enabled": True,
        "kohya_hrfix_block_number": "0:(1)",
        "kohya_hrfix_downscale_factor": "0:(2.0)",
        "kohya_hrfix_start_percent": "0:(0.0)",
        "kohya_hrfix_end_percent": "0:(0.35)",
        "kohya_hrfix_downscale_after_skip": True,
        "kohya_hrfix_downscale_method": "bicubic",
        "kohya_hrfix_upscale_method": "bicubic",
        "skip_video_creation": False,
        "fps": 20,
        "make_gif": False,
        "delete_imgs": False,
        "delete_input_frames": False,
        "add_soundtrack": "File",
        "soundtrack_path": "/home/rewbs/Rebound1.mp3",
        "r_upscale_video": False,
        "r_upscale_factor": "x2",
        "r_upscale_model": "realesr-animevideov3",
        "r_upscale_keep_imgs": True,
        "store_frames_in_ram": False,
        "frame_interpolation_engine": "RIFE",
        "frame_interpolation_x_amount": 2,
        "frame_interpolation_slow_mo_enabled": False,
        "frame_interpolation_slow_mo_amount": 0,
        "frame_interpolation_keep_imgs": True,
        "frame_interpolation_use_upscaled": False,
        "sd_model_name": "protovisionxl.safetensors",
        "sd_model_hash": "81b8e089",
        "deforum_git_commit_id": "Unknown"
    }

    gen = pipeline(**args)
    logger.info(f"Output video: {gen.video_path}")
