from qtpy.QtCore import Signal, QThread

from deforum import logger
from deforum.shared_storage import models

if "deforum_pipe" not in models:
    from deforum import DeforumAnimationPipeline
    models["deforum_pipe"] = DeforumAnimationPipeline.from_civitai(model_id="125703")

loaded_model_id = "125703"

class BackendThread(QThread):
    imageGenerated = Signal(object)  # Signal to emit the image data
    finished = Signal(dict)  # Signal to emit the image data

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
            # self.params['enable_subseed_scheduling'] = True
            # self.params['enable_steps_scheduling'] = True
            # self.params['color_coherence'] = False
            # self.params['hybrid_use_first_frame_as_init_image'] = False
            animation = models["deforum_pipe"](callback=datacallback, **self.params) if not use_settings_file else models["deforum_pipe"](callback=datacallback, settings_file=file_path)
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
