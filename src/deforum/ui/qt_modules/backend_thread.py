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
    models['deforum_pipe'] = DeforumAnimationPipeline.from_civitai(model_id="125703")#125703 424460
    # _ = models['deforum_pipe'](callback=None, max_frames=1, skip_save_video=True, store_frames_in_ram=True,
    #                               batch_name='temp', timestring='temp')

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
        # try:
        # from deforum.shared_storage import models
        # if loaded_model_id != self.params.get('model_id', "125703") and 'deforum_pipe' in models:
        #     try:
        #         models['deforum_pipe'].generator.cleanup()
        #     except:
        #         pass
        # # Load the deforum pipeline if not already loaded
        # if "deforum_pipe" not in models or loaded_model_id != self.params.get('model_id', "125703"):
        #     from deforum import DeforumAnimationPipeline
        #     models['deforum_pipe'] = DeforumAnimationPipeline.from_civitai(model_id=self.params.get('model_id', "125703"))
        #     loaded_model_id = self.params.get('model_id', "125703")
                                                                        #generator_name='DeforumDiffusersGenerator')
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
                            print("JOB DATA", job_data)
                            for k, v in job_data.items():
                                self.params[k] = v
                    except json.JSONDecodeError:
                        # Handle the case where the file is not valid JSON
                        print("Error: The file is not valid JSON.")
                    except FileNotFoundError:
                        # Handle the case where the file does not exist
                        print("Error: The file was not found.")
                else:
                    print("Error: The file is not a .txt file.")

        print(self.params)
        print(self.params['batch_name'])
        timestring = time.strftime('%Y%m%d%H%M%S')

        self.params['batch_name'] = f"{self.params['batch_name']}"
        self.params['timestring'] = timestring
        # self.params['enable_subseed_scheduling'] = True
        # self.params['enable_steps_scheduling'] = True
        # self.params['color_coherence'] = False
        # self.params['hybrid_use_first_frame_as_init_image'] = False

        self.should_wait = True
        if self.params.get('generate_viz'):
            # from deforum.utils.constants import config
            # config.allow_blocking_input_frame_lists = True
            import os
            from deforum.utils.constants import config
            output_path = os.path.join(config.root_path, 'output', 'deforum', f"{self.params['batch_name']}_{timestring}", 'inputframes')
            self.params['max_frames'] = int(math.floor(self.params['fps'] * get_audio_duration(self.params['audio_path'])) / self.params["extract_nth_frame"])
            os.makedirs(output_path, exist_ok=True)

            # anim = models['deforum_pipe'](callback=datacallback, max_frames=1, skip_save_video=True, store_frames_in_ram=True, batch_name='temp', timestring='temp')
            # time.sleep(2)
            # Define the base command
            base_command = "EGL_PLATFORM=surfaceless projectMCli"
            # Assemble the command with arguments
            command = f'{base_command} -a "{self.params["audio_path"]}" --presetFile "{os.path.join(config.root_path, "milks", self.params["milk_path"])}" --outputType image --outputPath "{str(output_path)}/" --fps 24 --width {self.params["width"]} --height {self.params["height"]}'

            print(command)

            self.process = subprocess.run(command, shell=True)
            if self.params["extract_nth_frame"] > 1:
                extract_nth_files(output_path, self.params["extract_nth_frame"])
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


        animation = models['deforum_pipe'](callback=datacallback, **self.params)
        result = {"status":"Ready",
                  "timestring":animation.timestring,
                  "resume_path":animation.outdir,
                  "resume_from":animation.image_count}
        if hasattr(animation, 'video_path'):
            result["video_path"] = animation.video_path
        print("Emitting", result)
        if self.process:
            del self.process  # Ensure process is terminated
            self.process = None
        self.finished.emit(result)
        # except Exception as e:
        #     logger.info(repr(e))
        #     self.finished.emit({"status": "Error"})
    def continueProcessing(self):
        pass
        # self.mutex.lock()
        # self.condition.wakeAll()
        # self.mutex.unlock()
