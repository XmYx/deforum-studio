import os
import subprocess
import time
import tempfile
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from deforum.utils.logging_config import logger


def save_as_gif(frames, filename):
    # Convert frames to gif
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # You can adjust this duration as needed
        loop=0,
    )

def save_as_h264(frames, filename, audio_path=None, fps=12):
    if len(frames) <= 0:
        logger.error("The buffer is empty, cannot save.")
        return

    # Handle frames input either as paths or as numpy arrays
    if isinstance(frames[0], str):
        frames = [Image.open(frame_path) for frame_path in frames]
    elif isinstance(frames[0], np.ndarray):
        frames = [Image.fromarray(frame) for frame in frames]
    writer = imageio.get_writer(filename, fps=fps, codec='libx264', pixelformat='yuv420p', output_params=['-crf', '17'])

    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()

    if audio_path:
        try:
            # audio_path could in fact be the init video so extract audio here. If audio_path 
            extracted_audio_tmpfile = tempfile.NamedTemporaryFile(suffix=".aac").name
            extract_cmd = ['ffmpeg', '-y', '-i', f'{audio_path}', '-vn', '-acodec', 'aac', '-strict', 'experimental', extracted_audio_tmpfile]
            logger.debug(f"Audio extraction command: {' '.join(extract_cmd)}")
            extract_process = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if extract_process.returncode != 0:
                logger.error(f"Audio extraction failed: {extract_process.stderr}")
                return

            # Calculate the duration of the video
            video_duration = len(frames) / fps

            # Merge the extracted audio with the video file
            output_filename = filename.replace(".mp4", "_with_audio.mp4")                
            merge_cmd = ['ffmpeg', '-y', '-i', filename, '-i', extracted_audio_tmpfile,
                            '-c:v', 'copy', '-c:a', 'copy', '-strict', 'experimental',
                            '-t', str(video_duration), output_filename]
            logger.debug(f"Audio merge command: {' '.join(merge_cmd)}")
            merge_process = subprocess.run(merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if merge_process.returncode == 0:
                os.rename(output_filename, filename)  # Replace the original file with the merged audio version
                os.remove(extracted_audio_tmpfile)  # Cleanup the extracted audio file
                logger.info("Audio merged successfully.")
            else:
                logger.error(f"Audio merging failed: {merge_process.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Audio processing failed: {e.stderr.decode('utf-8')}")
        except Exception as e:
            logger.error(f"An error occurred during audio merging: {str(e)}")

        finally:
            if os.path.exists(extracted_audio_tmpfile):
                os.remove(extracted_audio_tmpfile)

    return filename
