import subprocess
import time

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
    if len(frames) > 0:

        if isinstance(frames[0], np.ndarray):
            frames = [Image.fromarray(frame) for frame in frames]

        width = frames[0].size[0]
        height = frames[0].size[1]

        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}',
               '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-',
               '-c:v', 'libx264', '-profile:v', 'baseline', '-level', '3.0',
               '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23', '-an', filename]

        video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        for frame in tqdm(frames, desc="Saving MP4 (ffmpeg)"):
            frame_np = np.array(frame)  # Convert the PIL image to numpy array
            video_writer.stdin.write(frame_np.tobytes())

        _, stderr = video_writer.communicate()

        if video_writer.returncode != 0:
            logger.info(f"FFmpeg encountered an error: {stderr.decode('utf-8')}")
            return

        # if audio path is provided, merge the audio and the video
        if audio_path is not None:
            output_filename = f"output/mp4s/{time.strftime('%Y%m%d%H%M%S')}_with_audio.mp4"
            cmd = ['ffmpeg', '-y', '-i', filename, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', output_filename]

            result = subprocess.run(cmd, stderr=subprocess.PIPE)
            if result.returncode != 0:
                logger.info(f"Audio file merge failed from path {audio_path}\n{result.stderr.decode('utf-8')}")
    else:
        logger.info("The buffer is empty, cannot save.")
