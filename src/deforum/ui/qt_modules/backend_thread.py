import math
import time

from qtpy.QtCore import Signal, QThread

from deforum import logger
from deforum.shared_storage import models
from deforum.utils.audio_utils.deforum_audio import get_audio_duration

if "deforum_pipe" not in models:
    from deforum import DeforumAnimationPipeline
    models["deforum_pipe"] = DeforumAnimationPipeline.from_civitai(model_id="125703")

loaded_model_id = "125703"

class BackendThread(QThread):
    imageGenerated = Signal(object)  # Signal to emit the image data
    finished = Signal(dict)  # Signal to emit the image data
    generateViz = Signal(dict)


    def __init__(self, params):
        super().__init__()
        self.params = params
    def run(self):
        global loaded_model_id
        try:
            from deforum.shared_storage import models
            if loaded_model_id != self.params.get('model_id', "125703") and 'deforum_pipe' in models:
                try:
                    models["deforum_pipe"].generator.cleanup()
                except:
                    pass
            # Load the deforum pipeline if not already loaded
            if "deforum_pipe" not in models or loaded_model_id != self.params.get('model_id', "125703"):
                from deforum import DeforumAnimationPipeline
                models["deforum_pipe"] = DeforumAnimationPipeline.from_civitai(model_id=self.params.get('model_id', "125703"))
                loaded_model_id = self.params.get('model_id', "125703")
                                                                            #generator_name='DeforumDiffusersGenerator')
            models["deforum_pipe"].generator.optimize = self.params.get('optimize', True)
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
            timestring = time.strftime('%Y%m%d%H%M%S')
            self.params['batch_name'] = f"{self.params['batch_name']}"
            self.params['timestring'] = timestring
            # self.params['enable_subseed_scheduling'] = True
            # self.params['enable_steps_scheduling'] = True
            # self.params['color_coherence'] = False
            # self.params['hybrid_use_first_frame_as_init_image'] = False

            if self.params['generate_viz']:
                from deforum.utils.constants import config
                config.allow_blocking_input_frame_lists = True
                import os
                from deforum.utils.constants import root_path
                output_path = os.path.join(root_path, 'output', 'deforum', f"{self.params['batch_name']}_{timestring}", 'inputframes')
                self.params['max_frames'] = math.floor(self.params['fps'] * get_audio_duration(self.params['audio_path']))
                os.makedirs(output_path, exist_ok=True)
                self.generateViz.emit({"output_path":str(output_path),
                                       "fps":self.params['fps'],
                                       "width":self.params['width'],
                                       "height":self.params['height']})

            animation = models["deforum_pipe"](callback=datacallback, **self.params) if not use_settings_file else models["deforum_pipe"](callback=datacallback, batch_name=self.params['batch_name'], timestring=timestring, settings_file=file_path)
            result = {"status":"Ready",
                      "timestring":animation.timestring,
                      "resume_path":animation.outdir,
                      "resume_from":animation.image_count}
            if hasattr(animation, 'video_path'):
                result["video_path"] = animation.video_path
            self.finished.emit(result)
        except Exception as e:
            logger.info(repr(e))
            self.finished.emit({"status": "Error"})
