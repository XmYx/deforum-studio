import os
import subprocess
import time
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
    if len(frames) > 0:
        # Handle frames input either as paths or as numpy arrays
        if isinstance(frames[0], str):
            frames = [Image.open(frame_path) for frame_path in frames]
        elif isinstance(frames[0], np.ndarray):
            frames = [Image.fromarray(frame) for frame in frames]

        width, height = frames[0].size

        writer = imageio.get_writer(filename, fps=fps, codec='libx264',
                                    pixelformat='yuv420p', output_params=['-crf', '5'])

        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()

        # Extract and merge audio if an audio path is provided
        # audio_path = False
        if audio_path:
            logger.info("Audio Extraction not implemented with imageio yet, you need to have ffmpeg available in your system")
            try:
                extracted_audio_path = "extracted_audio.aac"
                # Extract the audio from the input video file
                extract_cmd = ['ffmpeg', '-y', '-i', audio_path, '-vn', '-acodec', 'copy', extracted_audio_path]
                subprocess.run(extract_cmd, stderr=subprocess.PIPE)

                # Calculate the duration of the video
                video_duration = len(frames) / fps
                output_filename = filename.replace(".mp4", "_with_audio.mp4")

                # Merge the extracted audio with the video file
                merge_cmd = ['ffmpeg', '-y', '-i', filename, '-i', extracted_audio_path,
                             '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
                             '-t', str(video_duration), output_filename]
                result = subprocess.run(merge_cmd, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    logger.error(f"Audio file merge failed: {result.stderr.decode('utf-8')}")
                else:
                    os.rename(output_filename, filename)  # Replace the original file with the merged audio version
                    os.remove(extracted_audio_path)  # Cleanup the extracted audio file
            except:
                logger.info("Audio merging failed")
            # extracted_audio_path = "extracted_audio.aac"
            # # Extract the audio from the input video file
            # imageio.ffmpeg_extract_audio(audio_path, extracted_audio_path)
            #
            # # Calculate the duration of the video
            # video_duration = len(frames) / fps
            # output_filename = filename.replace(".mp4", "_with_audio.mp4")
            #
            # # Merge the extracted audio with the video file
            # imageio.ffmpeg_merge_video_audio(filename, extracted_audio_path, output_filename,
            #                                  vcodec='copy', acodec='aac', ffmpeg_params=['-t', str(video_duration)])
            #
            # os.rename(output_filename, filename)  # Replace the original file with the merged audio version
            # os.remove(extracted_audio_path)  # Cleanup the extracted audio file
    else:
        logger.info("The buffer is empty, cannot save.")

# def save_as_h264(frames, filename, audio_path=None, fps=12):
#     if len(frames) > 0:
#         # Handle frames input either as paths or as numpy arrays
#         if isinstance(frames[0], str):
#             frames = [Image.open(frame_path) for frame_path in frames]
#         elif isinstance(frames[0], np.ndarray):
#             frames = [Image.fromarray(frame) for frame in frames]
#
#         width, height = frames[0].size
#
#         # Define command to create video from frames
#         cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}',
#                '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-profile:v', 'baseline',
#                '-level', '3.0', '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23', filename]
#         if audio_path:
#             cmd += ['-an']  # Temporarily disable audio in the first pass
#
#         # Execute ffmpeg to create the video
#         video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
#         for frame in tqdm(frames, desc="Saving MP4 (ffmpeg)"):
#             video_writer.stdin.write(np.array(frame).tobytes())
#
#         _, stderr = video_writer.communicate()
#
#         if video_writer.returncode != 0:
#             logger.error(f"FFmpeg encountered an error: {stderr.decode('utf-8')}")
#             return
#
#         # Extract and merge audio if an audio path is provided
#         if audio_path:
#             extracted_audio_path = "extracted_audio.aac"
#             # Extract the audio from the input video file
#             extract_cmd = ['ffmpeg', '-y', '-i', audio_path, '-vn', '-acodec', 'copy', extracted_audio_path]
#             subprocess.run(extract_cmd, stderr=subprocess.PIPE)
#
#             # Calculate the duration of the video
#             video_duration = len(frames) / fps
#             output_filename = filename.replace(".mp4", "_with_audio.mp4")
#
#             # Merge the extracted audio with the video file
#             merge_cmd = ['ffmpeg', '-y', '-i', filename, '-i', extracted_audio_path,
#                          '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
#                          '-t', str(video_duration), output_filename]
#             result = subprocess.run(merge_cmd, stderr=subprocess.PIPE)
#
#             if result.returncode != 0:
#                 logger.error(f"Audio file merge failed: {result.stderr.decode('utf-8')}")
#             else:
#                 os.rename(output_filename, filename)  # Replace the original file with the merged audio version
#                 os.remove(extracted_audio_path)  # Cleanup the extracted audio file
#     else:
#         logger.info("The buffer is empty, cannot save.")