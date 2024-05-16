"""
The given python script file is the CLI interface for DeForUM, allowing the interaction with defined DeForUM operational modes via terminal commands. Moreover, it includes methods for initiating web-based interfaces, animation processing, or running predefined presets. Additional provisions for setting up the environment and calling specific functionalities are also highlighted within the code.

First, the main docstring introduces a global CLI script that starts DeForUM with various operational modes based on the user's input. Then, functions in the script are explained:

1. `install_qtpy()`: This function ensures the proper installation of the QtPy library for UI-related activities by checking its presence; if not available, it installs PyQt6-Qt6, PyQt6 and QtPy via pip.

2. `start_deforum_cli() -> None`: This function serves as the main entry point for the command-line interface (CLI) through which users interact with DeForUM. Supported operational modes include 'webui', 'animatediff',  'runpresets', 'api', 'setup', 'ui', 'runsingle', 'config', 'unittest'. According to the mode selection, the corresponding operations are invoked.

Example Usage:

To start DeForUM in 'webui' mode, users can input this command in their terminal:
```shell
python deforum_cli.py webui
```

To guide DeForUM to run presets with a model name or id in 'runpresets' mode, use:
```shell
python deforum_cli.py runpresets --modelid 125703
```

To operate DeForUM in 'api' mode which starts a FastAPI, type:
```shell
python deforum_cli.py api
```

Raises:
    subprocess.CalledProcessError: If a subprocess for setting up the environment fails.
    ValueError: If an invalid option is passed to the command line arguments.
"""
import argparse
import os
import random
import subprocess
import sys
import traceback

from deforum.commands.deforum_run_unit_test import run_unit_test
from deforum.docutils.decorator import deforumdoc
from deforum.utils.constants import config
from deforum.utils.logging_config import logger
@deforumdoc
def install_qtpy() -> None:
    """
    Function to install the qtpy package using pip if it is not already installed.

    This function tries to import the qtpy module. If it fails, the function uses subprocess to run pip install
    commands for the PyQt6-Qt6, pyqt6, and qtpy packages.

    Raises:
        subprocess.CalledProcessError: If the pip installation subprocess fails.
    """
    try:
        import qtpy
    except:
        subprocess.run(
            [
                "python3",
                "-m" "pip",
                "install",
                "PyQt6-Qt6==6.5.0",
            ]
        )
        subprocess.run(
            [
                "python3",
                "-m" "pip",
                "install",
                "pyqt6==6.5.0",
            ]
        )
        subprocess.run(
            [
                "python3",
                "-m" "pip",
                "install",
                "qtpy==2.4.1",
            ]
        )
        subprocess.run(
            [
                "python3",
                "-m" "pip",
                "install",
                "pyqtgraph==0.13.7",
            ]
        )
@deforumdoc
def start_deforum_cli() -> None:
    """
    Function to start the DeForUM's Consule Line Interface, featuring various operational modes.

    This function parses command line arguments to determine the operational mode of DeForUM. The operational
    modes include 'webui', 'animatediff', 'runpresets', 'api', 'setup', 'ui', 'runsingle', 'config', and 'unittest'.
    Depending on the selected mode, different components of the DeForUM suite are invoked. A file path and additional
    options can also be passed as arguments to the function.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Load settings from a txt file and run the deforum process.")
    # Positional mode argument
    parser.add_argument("mode", choices=['webui', 'animatediff', 'runpresets', 'api', 'setup', 'ui', 'runsingle', 'config', 'unittest', 'version'], default=None, nargs='?',
                        help="Choose the mode to run.")

    parser.add_argument("--file", type=str, help="Path to the deforum settings file.")
    parser.add_argument("--options", nargs=argparse.REMAINDER,
                        help="Additional keyword arguments to pass to the deforum function.")
    args_main = parser.parse_args()

    extra_args = {}

    if args_main.file:
        extra_args["settings_file"] = args_main.file

    def convert_value(in_value):
        """Converts the string value to its corresponding data type."""
        # Check if this is a list of values separated by commas
        if '|' in in_value and not in_value.startswith('"'):
            return in_value.split('|')

        if in_value.isdigit():
            return int(in_value)
        try:
            return float(in_value)
        except ValueError:
            if in_value.startswith('"') and in_value.endswith('"'):
                # Remove the quotes and return the string or a list
                in_value = in_value[1:-1]
                if '|' in in_value:  # It's a list of strings
                    return in_value.split('|')
                else:
                    return in_value  # It's a single string
            else:
                return in_value

    options = {}
    if args_main.options:
        for item in args_main.options:
            key, value_str = item.split('=')
            value = convert_value(value_str)
            options[key] = value

    if args_main.mode:
        if args_main.mode == "version":
            import deforum
            print(deforum.__version__)
            logger.info(str(deforum.__version__))
        elif args_main.mode == "webui":
            import streamlit.web.cli as stcli
            stcli.main(["run", f"{config.src_path}/deforum/webui/deforum_webui.py", "--server.headless", "true"])
        elif args_main.mode == "animatediff":
            from deforum.pipelines.animatediff_animation.pipeline_animatediff_animation import DeforumAnimateDiffPipeline
            modelid = str(options.get("modelid", "132632"))
            pipe = DeforumAnimateDiffPipeline.from_civitai(model_id=modelid)
            _ = pipe(**extra_args, **options)

        elif args_main.mode == "runpresets":
            from deforum import DeforumAnimationPipeline
            modelid = str(options.get("modelid", "125703"))

            deforum = DeforumAnimationPipeline.from_civitai(model_id=modelid)

            preset_dir = "presets"
            txt_files = [os.path.join(root, file)
                        for root, _, files in os.walk(preset_dir)
                        for file in files if file.endswith(".txt")]

            if options.get("randomize_files", False):
                random.shuffle(txt_files)

            for file_path in txt_files:
                try:
                    logger.info(f"Settings file path: {file_path}")
                    batch_name = os.path.splitext(os.path.basename(file_path))[0]
                    logger.info(f"Batch Name: {batch_name}")

                    extra_args["settings_file"] = file_path

                    options.update({
                        "prompts": {"0": "A solo delorean speeding on an ethereal highway through time jumps, like in the iconic movie back to the future."},
                        "seed": 420,
                    })

                    deforum(**extra_args, **options)
                except Exception as e:
                    logger.error(traceback.format_exc())

        elif args_main.mode == "api":
            from fastapi import FastAPI, WebSocket
            import uvicorn
            from deforum import DeforumAnimationPipeline
            modelid = str(options.get("modelid", "125703"))

            deforum = DeforumAnimationPipeline.from_civitai(model_id=modelid)
            app = FastAPI()

            async def image_callback(websocket, image):
                await websocket.send_text(image)
            @app.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                await websocket.accept()
                data = await websocket.receive_json()

                # Assuming data contains the necessary information for deforum to run
                async def callback(data):
                    logger.info("deforum callback")
                    image = data.get('image')
                    if image is not None:
                        await websocket.send_text("image")


                video_path = await run_deforum(data, callback)
                # After deforum processing, send the video_path
                if video_path is not None:
                    await websocket.send_text(video_path)
                await websocket.close()

            async def run_deforum(data, callback):
                # Set up and run the deforum process, using the callback to send images
                # For demonstration, let's assume DeforumAnimationPipeline has a method `run_with_callback`
                deforum.datacallback = callback
                animation = await deforum(**data)
                if hasattr(animation, "video_path"):
                    return animation.video_path
                else:
                    return None

            # Start the Uvicorn server
            uvicorn.run(app, host="localhost", port=8000)
        elif args_main.mode == "setup":
            logger.info("Installing stable-fast and its dependencies...")
            from deforum.utils.install_sfast import install_sfast
            install_sfast()
        elif args_main.mode == "ui":
            install_qtpy()

            # Get the absolute path of the current file
            current_file_path = os.path.abspath(__file__)

            # Get the parent directory of the current file
            parent_directory = os.path.dirname(current_file_path)

            # Assuming 'deforum' is in the parent directory of the current file
            deforum_directory = os.path.dirname(parent_directory)
            # Construct the path to main.py
            main_script_path = os.path.join(deforum_directory, "ui", "main.py")
            # try:
            # Execute main.py
            subprocess.run([sys.executable, main_script_path])
        elif args_main.mode == 'runsingle':
            install_qtpy()

            # Get the absolute path of the current file
            current_file_path = os.path.abspath(__file__)

            # Get the parent directory of the current file
            parent_directory = os.path.dirname(current_file_path)

            # Assuming 'deforum' is in the parent directory of the current file
            deforum_directory = os.path.dirname(parent_directory)
            # Construct the path to main.py
            logger.info(f"Using settings file: {extra_args['settings_file']}")

            main_script_path = os.path.join(deforum_directory, "ui", "process_only.py")
            # try:
            # Execute main.py
            subprocess.run([sys.executable, main_script_path, f"{extra_args['settings_file']}"])
        elif args_main.mode == 'config':
            # Get the absolute path of the current file
            current_file_path = os.path.abspath(__file__)

            # Get the parent directory of the current file
            parent_directory = os.path.dirname(current_file_path)

            # Assuming 'deforum' is in the parent directory of the current file
            deforum_directory = os.path.dirname(parent_directory)
            # Construct the path to main.py
            main_script_path = os.path.join(deforum_directory, "commands", "deforum_config.py")
            subprocess.run([sys.executable, main_script_path])
        elif args_main.mode == "unittest":
            run_unit_test(options, extra_args)
    else:
        from deforum import DeforumAnimationPipeline
        deforum = DeforumAnimationPipeline.from_civitai()
        deforum.generator.optimize = False
        gen = deforum(**extra_args, **options)
        logger.info(f"Output video: {gen.video_path} ")