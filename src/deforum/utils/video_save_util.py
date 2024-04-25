import os
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

        width, height = frames[0].size

        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}',
               '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-profile:v', 'baseline',
               '-level', '3.0', '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23', filename]
        if audio_path:
            cmd += ['-an']  # Temporarily disable audio in the first pass

        video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        for frame in tqdm(frames, desc="Saving MP4 (ffmpeg)"):
            video_writer.stdin.write(np.array(frame).tobytes())

        _, stderr = video_writer.communicate()

        if video_writer.returncode != 0:
            logger.error(f"FFmpeg encountered an error: {stderr.decode('utf-8')}")
            return

        # Merge audio if an audio path is provided
        if audio_path:
            # Calculate the duration of the video
            video_duration = len(frames) / fps
            output_filename = filename.replace(".mp4", "_with_audio.mp4")
            cmd = ['ffmpeg', '-y', '-i', filename, '-stream_loop', '-1', '-i', audio_path,
                   '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
                   '-t', str(video_duration), output_filename]  # Use the `-t` flag to match video duration
            result = subprocess.run(cmd, stderr=subprocess.PIPE)
            if result.returncode != 0:
                logger.error(f"Audio file merge failed: {result.stderr.decode('utf-8')}")
            else:
                os.rename(output_filename, filename)  # Replace the original file with the merged audio version
    else:
        logger.info("The buffer is empty, cannot save.")