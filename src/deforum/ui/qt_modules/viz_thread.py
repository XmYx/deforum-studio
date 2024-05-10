import tempfile
import time

from qtpy.QtCore import QThread, Signal

# class VisualGeneratorThread(QThread):
#     # Signal to emit when processing is potentially finished
#     finished = Signal(dict)
#
#     def __init__(self, audio_path, output_path, preset_path, fps, width, height):
#         super().__init__()
#         self.audio_path = audio_path
#         self.output_path = output_path
#         self.preset_path = preset_path
#         self.fps = fps
#         self.width = width
#         self.height = height
#
#     def run(self):
#         """
#         Starts the process to generate a visual representation of an audio file in a new thread.
#         """
#         # Ensure the output directory exists
#         os.makedirs(self.output_path, exist_ok=True)
#
#         # Define the base command
#         base_command = "EGL_PLATFORM=surfaceless projectMCli"
#
#         # Assemble the command with arguments
#         command = f'{base_command} -a "{self.audio_path}" --presetFile "{self.preset_path}" --outputType image --outputPath "{self.output_path}/" --fps 24 --width {self.width} --height {self.height}'
#
#         # Start the command in a non-blocking subprocess
#         # Start the command in a non-blocking subprocess
#         try:
#             viz_process = subprocess.run(command, shell=True)
#             del viz_process
#             viz_process = None
#
#             from deforum.utils.constants import root_path
#             self.temp_video_path = os.path.join(root_path, 'temp_video.mp4')
#             os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
#
#             images_folder = Path(self.output_path)
#             image_files = sorted(images_folder.glob('*.jpg'), key=lambda x: int(x.stem))
#             writer = imageio.get_writer(self.temp_video_path, fps=self.fps)
#
#             for image_path in image_files:
#                 image = imageio.imread(image_path)
#                 writer.append_data(image)
#             writer.close()
#
#             self.output_path = os.path.join(root_path, 'output.mp4')
#
#             ffmpeg_command = [
#                 'ffmpeg', '-y',
#                 '-i', self.temp_video_path,
#                 '-i', self.audio_path,
#                 '-c:v', 'copy',
#                 '-c:a', 'aac',
#                 '-strict', 'experimental',
#                 '-shortest',
#                 self.output_path
#             ]
#
#             video_process = subprocess.run(ffmpeg_command, text=True)
#
#             del video_process
#             video_process = None
#             self.finished.emit({'video_path': self.output_path})
#             # Check if the process was successful
#
#             # self.finished.emit(self.output_path)  # Emit the output path if successful
#
#
#         except Exception as e:
#             print("Error starting the command:", str(e))
#
# #     # Signal to emit when processing is finished
# #     finished = Signal(str)
# #
# #     def __init__(self, audio_path, output_path, preset_path, fps, width, height):
# #         super().__init__()
# #         self.audio_path = audio_path
# #         self.output_path = output_path
# #         self.preset_path = preset_path
# #         self.fps = fps
# #         self.width = width
# #         self.height = height
# #     def run(self):
# #         """
# #         Runs the command to generate a visual representation of an audio file in a new thread.
# #         """
# #         # Ensure the output directory exists
# #         os.makedirs(self.output_path, exist_ok=True)
# #
# #         # Define the base command
# #         base_command = "EGL_PLATFORM=surfaceless projectMCli"
# #
# #         # Assemble the command with arguments
# #         command = f'{base_command} -a "{self.audio_path}" --presetFile "{self.preset_path}" --outputType image --outputPath "{self.output_path}/" --fps {self.fps} --width {self.width} --height {self.height}'
# #
# #         # Execute the command in a non-blocking subprocess
# #         try:
# #             process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# #             stdout, stderr = process.communicate()  # This waits for the process to finish
# #
# #             if process.returncode == 0:
# #                 print("Command executed successfully:", stdout.decode())
# #                 self.finished.emit(self.output_path)  # Emit the finished signal with the output path
# #             else:
# #                 print("Error in command execution:", stderr.decode())
# #                 self.finished.emit('')  # Emit empty string or handle errors differently
# #
# #         except Exception as e:
# #             print("Error starting the command:", str(e))
# #             self.finished.emit('')  # Emit empty string in case of exception
#
# # class VideoAssemblerThread(QThread):
# #     # Signal to emit when video assembly is finished
# #     finished = Signal(dict)
# #
# #     def __init__(self, images_folder, fps, output_path):
# #         super().__init__()
# #         self.images_folder = images_folder
# #         self.fps = fps
# #         # self.output_path = output_path
# #
# #     def run(self):
# #         """
# #         Assembles images into a video at the specified fps using imageio-ffmpeg.
# #         """
# #         try:
# #             from deforum.utils.constants import root_path
# #             self.output_path = os.path.join(root_path, 'temp.mp4')
# #             # Ensure the output directory exists
# #             # os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
# #
# #             # List all PNG images in the folder and sort them by the numeric part in the filename
# #             images_folder = Path(self.images_folder)
# #             image_files = sorted(images_folder.glob('*.jpg'), key=lambda x: int(x.stem))
# #
# #             # Create a video writer object
# #             writer = imageio.get_writer(self.output_path, fps=self.fps)
# #
# #             # Write frames to video
# #             for image_path in image_files:
# #                 image = imageio.imread(image_path)
# #                 writer.append_data(image)
# #
# #             # Close the writer to finish writing the video file
# #             writer.close()
# #
# #             # Check if video was created successfully
# #             if os.path.exists(self.output_path):
# #                 print("Video assembled successfully.")
# #                 self.finished.emit({'video_path': self.output_path})  # Emit the finished signal with the video file path
# #             else:
# #                 print("Failed to create video.")
# #                 self.finished.emit({})  # Emit empty dict or handle errors differently
# #
# #         except Exception as e:
# #             print("Error in assembling video:", str(e))
# #             self.finished.emit({})  # Emit empty dict in case of exception
#
# class VideoAssemblerThread(QThread):
#     finished = Signal(dict)
#
#     def __init__(self, images_folder, fps, audio_path):
#         super().__init__()
#         self.images_folder = images_folder
#         self.fps = fps
#         self.audio_path = audio_path
#
#     def run(self):
#         try:
#             print("ASSEMBLER CALLED NOW")
#             from deforum.utils.constants import root_path
#             self.output_path = os.path.join(root_path, 'output.mp4')
#             self.temp_video_path = os.path.join(root_path, 'temp_video.mp4')
#             os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
#
#             images_folder = Path(self.images_folder)
#             image_files = sorted(images_folder.glob('*.jpg'), key=lambda x: int(x.stem))
#             writer = imageio.get_writer(self.temp_video_path, fps=self.fps)
#
#             for image_path in image_files:
#                 image = imageio.imread(image_path)
#                 writer.append_data(image)
#             writer.close()
#
#             ffmpeg_command = [
#                 'ffmpeg', '-y',
#                 '-i', self.temp_video_path,
#                 '-i', self.audio_path,
#                 '-c:v', 'copy',
#                 '-c:a', 'aac',
#                 '-strict', 'experimental',
#                 '-shortest',
#                 self.output_path
#             ]
#
#             process = subprocess.run(ffmpeg_command, text=True)
#             if process.returncode == 0:
#                 self.finished.emit({'video_path': self.output_path})
#
#
#         except Exception as e:
#             print(f"Error in assembling video: {str(e)}")
#
#
#
#
# # png_files = [f for f in images_folder.glob('*.png')]
# #
# # # Sort the files by creation time
# # image_files = list(sorted(png_files, key=lambda x: x.stat().st_ctime))
# # # image_files.sort(key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group()))
# #
# # # Generate a sorted file list for ffmpeg
# # sorted_file_list = os.path.join(self.images_folder, "sorted_files.txt")
# # with open(sorted_file_list, "w") as file:
# #     for image in image_files:
# #         file.write(f"file '{image}'\n")
# #
# # # Define the ffmpeg command to assemble the video
# # command = (
# #     f"ffmpeg -y -f concat -safe 0 -i {sorted_file_list} "
# #     f"-c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p "
# #     f"-preset fast -crf 23 -movflags +faststart {video_file_path}"
# # )
import os
import subprocess
from pathlib import Path
import imageio
from multiprocessing import Process, Queue

from deforum.utils.constants import root_path


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
        temp_path = os.path.join(self.output_path, 'temp')
        os.makedirs(temp_path, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=temp_path)
        self.temp_video_path = temp_file.name
        temp_file.close()

        # queue = Queue()
        # process = Process(target=self.run_subprocess, args=(self.build_viz_command(), queue))
        # process.start()
        # process.join()  # Ensure the process has finished

        subprocess.run(self.build_viz_command(), shell=True)
        # self.run_subprocess(self.build_viz_command())

        # Wait for a small delay to allow file system updates (if necessary)
        # time.sleep(0.5)

        # Check if the video file was actually created
        if os.path.exists(self.temp_video_path):
            print("Video generated:", self.temp_video_path)
            self.finished.emit({'video_path': self.temp_video_path})
        else:
            print("Failed to generate video.")
            self.finished.emit({'error': 'Failed to generate video.'})

    def build_viz_command(self):

        print("MILK:", self.preset_path)

        base_command = "EGL_PLATFORM=surfaceless projectMCli"
        return f'{base_command} -a "{self.audio_path}" --presetFile "{self.preset_path}" --outputType video --outputPath "{self.temp_video_path}" --fps 24 --width {self.width} --height {self.height}'

    def run_subprocess(self, command, queue):
        try:
            proc = subprocess.run(command, shell=True, text=True, capture_output=True)
            print(proc.stderr, proc.stdout)
            if proc.returncode != 0:
                print(f"Subprocess ended with return code {proc.returncode}")
            queue.put('finished')
        except Exception as e:
            print(f"Error in subprocess: {e}")
            queue.put('error')

    def compile_images_to_video(self, queue):
        try:

            temp_path = os.path.join(root_path, 'temp')
            os.makedirs(temp_path, exist_ok=True)

            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False,
                                                    dir=temp_path)

            # We get the path and then close the file handle. The file itself remains.
            self.temp_video_path = temp_file.name
            temp_file.close()
            os.remove(self.temp_video_path)

            # temp_video_path = os.path.join(self.output_path, 'temp', 'temp_video.mp4')
            images_folder = Path(self.output_path)
            image_files = sorted(images_folder.glob('*.jpg'), key=lambda x: int(x.stem))
            with imageio.get_writer(self.temp_video_path, fps=24) as writer:
                for image_path in image_files:
                    image = imageio.imread(image_path)
                    writer.append_data(image)
            queue.put('success')
        except Exception as e:
            print(f"Error compiling images: {e}")
            queue.put('failure')

    def combine_audio_with_video(self, queue):
        try:
            temp_path = os.path.join(root_path, 'temp')
            os.makedirs(temp_path, exist_ok=True)

            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False,
                                                    dir=temp_path)

            # We get the path and then close the file handle. The file itself remains.
            temp_video_path = temp_file.name
            temp_file.close()
            os.remove(temp_video_path)

            # temp_video_path = os.path.join(self.output_path, 'temp_video.mp4')
            #output_path = os.path.join(self.output_path, 'output.mp4')
            ffmpeg_command = [
                'ffmpeg', '-y',
                '-i', self.temp_video_path,
                '-i', self.audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-shortest',
                temp_video_path
            ]
            proc = subprocess.run(ffmpeg_command)
            self.result_path = temp_video_path
            # proc.wait(timeout=300)  # Optional timeout in seconds
            # if proc.poll() is None:  # Check if process has finished
            #     proc.kill()  # Force kill if still running
            queue.put('success')
        except Exception as e:
            print(f"Error combining audio and video: {e}")
            queue.put('failure')