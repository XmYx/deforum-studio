import argparse
import os
import random
import subprocess
import sys

from deforum.utils.logging_config import logger

def install_pyqt6():
    try:
        import PyQt6
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

def start_deforum_cli():

    parser = argparse.ArgumentParser(description="Load settings from a txt file and run the deforum process.")
    # Positional mode argument
    parser.add_argument("mode", choices=['webui', 'animatediff', 'runpresets', 'api', 'setup', 'ui'], default=None, nargs='?',
                        help="Choose the mode to run.")

    parser.add_argument("--file", type=str, help="Path to the txt file containing dictionaries to merge.")
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
        if args_main.mode == "webui":
            import streamlit.web.cli as stcli
            root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            stcli.main(["run", f"{root_path}/webui/deforum_webui.py", "--server.headless", "true"])
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
            files = []
            for root, _, filenames in os.walk(preset_dir):
                for file in filenames:

                    files.append(os.path.join(root, file))

            if options.get("randomize_files", False):
                random.shuffle(files)

            for file_path in files:
                try:
                    logger.info(f"Settings file path: {file_path}")

                    batch_name = file_path.split('.')[0].split("/")[-1]
                    logger.info(f"Batch Name: {batch_name}")

                    extra_args["settings_file"] = file_path

                    options["prompts"] = {"0": "A solo delorean speeding on an ethereal highway through time jumps, like in the iconic movie back to the future."}
                    options["seed"] = 420
                    options["batch_name"] = batch_name

                    deforum(**extra_args, **options)
                except Exception as e:
                    logger.error(f"Error running settings file: {file_path}")

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
                    print("deforum callback")
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
            print("SETUP")
            from deforum.utils.install_sfast import install_sfast
            install_sfast()
        elif args_main.mode == "ui":
            install_pyqt6()

            # Get the absolute path of the current file
            current_file_path = os.path.abspath(__file__)

            # Get the parent directory of the current file
            parent_directory = os.path.dirname(current_file_path)

            # Assuming 'deforum' is in the parent directory of the current file
            deforum_directory = os.path.dirname(parent_directory)
            # Construct the path to main.py
            main_script_path = os.path.join(deforum_directory, "ui", "main.py")
            try:
            # Execute main.py
                subprocess.run([sys.executable, main_script_path])
            except:

                subprocess.run(
                    [
                        "python3",
                        "-m" "pip",
                        "uninstall",
                        "PyQt6-Qt6",
                        "-y"
                    ]
                )
                subprocess.run(
                    [
                        "python3",
                        "-m" "pip",
                        "install",
                        "PyQt6-Qt6==6.5.0",
                    ]
                )
                subprocess.run([sys.executable, main_script_path])

    else:
        from deforum import DeforumAnimationPipeline
        deforum = DeforumAnimationPipeline.from_civitai()
        _ = deforum(**extra_args, **options)
