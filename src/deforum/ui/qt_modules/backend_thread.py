import json
import math
import subprocess
import time
from pathlib import Path
from threading import Event

import imageio
from qtpy.QtCore import QWaitCondition, QMutex
from qtpy.QtCore import Signal, QThread

from deforum import logger
from deforum.shared_storage import models
from deforum.utils.audio_utils.deforum_audio import get_audio_duration

from deforum import DeforumAnimationPipeline
from deforum.shared_storage import models
from deforum.utils.file_utils.extract_nth_files import extract_nth_files

if 'deforum_pipe' not in models:
    models['deforum_pipe'] = DeforumAnimationPipeline.from_civitai(model_id="125703")

loaded_model_id = "125703"

class BackendThread(QThread):
    imageGenerated = Signal(object)  # Signal to emit the image data
    finished = Signal(dict)  # Signal to emit the image data
    generateViz = Signal(str)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.process = None
        # self.mutex = QMutex()
        # self.condition = QWaitCondition()

    def run(self):
        global loaded_model_id
        models['deforum_pipe'].generator.optimize = self.params.get('optimize', True)
        prom = self.params.get('prompts', 'cat sushi')
        key = self.params.get('keyframes', '0')
        if prom == "":
            prom = "Abstract art"
        if key == "":
            key = "0"

        if not isinstance(prom, dict):
            new_prom = list(prom.split("\n"))
            new_key = list(key.split("\n"))
            self.params["animation_prompts"] = dict(zip(new_key, new_prom))
        else:
            self.params["animation_prompts"] = prom

        # Call the deforum animation pipeline
        def datacallback(data):
            self.imageGenerated.emit(data)  # Emit the image data when available

        use_settings_file = False
        if 'settings_file' in self.params:
            file_path = self.params.pop('settings_file')
            if file_path:
                use_settings_file = True
                # Check if the file has a .txt extension
                if file_path.endswith('.txt'):
                    try:
                        with open(file_path, 'r') as file:
                            # Attempt to load the JSON data
                            job_data = json.load(file)
                            for k, v in job_data.items():
                                self.params[k] = v
                    except json.JSONDecodeError:
                        # Handle the case where the file is not valid JSON
                        logger.info("Error: The file is not valid JSON.")
                    except FileNotFoundError:
                        # Handle the case where the file does not exist
                        logger.info("Error: The file was not found.")
                else:
                    logger.info("Error: The file is not a .txt file.")

        timestring = time.strftime('%Y%m%d%H%M%S')

        self.params['batch_name'] = f"{self.params['batch_name']}"
        self.params['timestring'] = timestring if not self.params['resume_from_timestring'] else self.params['resume_timestring']

        self.should_wait = True
        if self.params.get('generate_viz'):
            import os
            from deforum.utils.constants import config
            output_path = os.path.join(config.root_path, 'output', 'deforum', f"{self.params['batch_name']}_{timestring}", 'inputframes')
            self.params['max_frames'] = int(math.floor(self.params['fps'] * get_audio_duration(self.params['audio_path'])) / self.params["extract_nth_frame"])
            os.makedirs(output_path, exist_ok=True)
            base_command = "EGL_PLATFORM=surfaceless projectMCli"
            # Assemble the command with arguments
            command = f'{base_command} -a "{self.params["audio_path"]}" --presetFile "{os.path.join(config.root_path, "milks", self.params["milk_path"])}" --outputType image --outputPath "{str(output_path)}/" --fps 24 --width {self.params["width"]} --height {self.params["height"]}'

            self.process = subprocess.run(command, shell=True)
            if self.params["extract_nth_frame"] > 1:
                _ = extract_nth_files(output_path, self.params["extract_nth_frame"])
            self.output_path = os.path.join(config.root_path, 'output.mp4')
            self.temp_video_path = os.path.join(config.root_path, 'temp_video.mp4')
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

            images_folder = Path(str(output_path))
            image_files = sorted(images_folder.glob('*.jpg'), key=lambda x: int(x.stem))

            writer = imageio.get_writer(self.temp_video_path, fps=24)

            for image_path in image_files:
                image = imageio.imread(image_path)
                writer.append_data(image)
            writer.close()

            ffmpeg_command = [
                'ffmpeg', '-y',
                '-i', self.temp_video_path,
                '-i', self.params["audio_path"],
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-shortest',
                self.output_path
            ]

            process = subprocess.run(ffmpeg_command, text=True)
            self.finished.emit({'video_path': self.output_path})

        if self.params['enable_ad_pass']:
            self.params['adiff_pass_params'] = {
                "max_frames": self.params['ad_max_frames'],
                "use_every_nth": self.params['ad_use_every_nth'],
                "width": self.params['width'],
                "height": self.params['height'],
                "closed_loop": self.params['ad_closed_loop'],
                "prompt": self.params['ad_prompt'],
                "negative_prompt": self.params['ad_negative_prompt'],
                "hires_steps": self.params['ad_hires_steps'],
                "hires_pass_denoise": 0.56,
                "controlnet_strength": self.params["ad_controlnet_strength"],
                "controlnet_start": 0.0,
                "controlnet_end": 0.5,
                "ip_adapter_video_strength": self.params["ad_ip_adapter_video_strength"],
                "ip_adapter_video_start": 0.0,
                "ip_adapter_video_end": 0.8,
                "ip_adapter_image_strength": self.params["ad_ip_adapter_image_strength"],
                "ip_adapter_image_start": 0.0,
                "ip_adapter_image_end": 0.5,
                "seed": self.params["ad_seed"],
                "steps": self.params["ad_steps"],
                "cfg": self.params["ad_cfg"],
                "sampler": self.params["ad_sampler_name"],
                "scheduler": self.params["ad_scheduler"],
                "start_at_step": self.params["ad_start_step"],
                "fps": self.params['fps'],
                "sd_model":self.params["ad_sd_model"],
                "lora":self.params["ad_lora"],
                "ad_model":self.params["ad_model"],
                "sampler_name": self.params["ad_sampler_name"],
                "beta_schedule": self.params["ad_beta_schedule"],
            }
        animation = models['deforum_pipe'](callback=datacallback, **self.params)
        result = {"status":"Ready",
                  "timestring":animation.timestring,
                  "resume_path":animation.outdir,
                  "resume_from":animation.max_frames}
        if hasattr(animation, 'video_path'):
            result["video_path"] = animation.video_path
        if self.process:
            del self.process  # Ensure process is terminated
            self.process = None
        self.finished.emit(result)
    def continueProcessing(self):
        pass
