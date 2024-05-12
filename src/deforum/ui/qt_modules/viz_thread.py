import tempfile
import os
import subprocess

from qtpy.QtCore import QThread, Signal
from deforum.utils.constants import config


class VisualGeneratorThread(QThread):
    finished = Signal(dict)

    def __init__(self, audio_path, output_path, preset_path, fps, width, height):
        super().__init__()
        self.audio_path = audio_path
        self.output_path = output_path
        self.preset_path = preset_path
        self.fps = fps
        self.width = width
        self.height = height

    def run(self):
        temp_path = os.path.join(config.root_path, 'temp')
        os.makedirs(temp_path, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=temp_path)
        self.temp_video_path = temp_file.name
        temp_file.close()
        os.remove(self.temp_video_path)

        subprocess.run(self.build_viz_command(), shell=True)
        self.finished.emit({'video_path': self.temp_video_path})

    def build_viz_command(self):

        base_command = "EGL_PLATFORM=surfaceless projectMCli"
        return f'{base_command} -a "{self.audio_path}" --presetFile "{self.preset_path}" --outputType video --outputPath "{self.temp_video_path}" --fps 24 --width {self.width} --height {self.height}'

