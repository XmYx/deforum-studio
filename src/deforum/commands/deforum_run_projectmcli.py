import os
import subprocess

def generate_visual_from_audio(audio_path, output_path, preset_path, fps, width, height):
    """
    Generates a visual representation of an audio file using projectM with a specified preset.
    This function runs the command in a non-blocking way.

    Args:
    audio_path (str): The file path of the input audio file.
    output_path (str): The directory path where the output image will be saved.
    preset_path (str): The file path of the milk preset used for visualization.

    Returns:
    None
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Define the base command
    base_command = "EGL_PLATFORM=surfaceless projectMCli"

    # Assemble the command with arguments
    command = f'{base_command} -a "{audio_path}" --presetFile "{preset_path}" --outputType image --outputPath "{output_path}/" --fps {fps} --width {width}, --height {height}'

    # Execute the command in a non-blocking subprocess
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Optional: Add logic here if you want to handle process output or wait for its completion in another way
        # For example, to print outputs asynchronously or handle them after certain other actions:
        # stdout, stderr = process.communicate()
        # print("Command output:", stdout.decode())
        # if process.returncode != 0:
        #     print("Error in command execution:", stderr.decode())

    except Exception as e:
        print("Error starting the command:", str(e))