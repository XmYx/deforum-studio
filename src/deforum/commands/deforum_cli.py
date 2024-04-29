import argparse
import os

def start_deforum_cli():

    parser = argparse.ArgumentParser(description="Load settings from a txt file and run the deforum process.")
    # Positional mode argument
    parser.add_argument("mode", choices=['webui', 'animatediff', 'runpresets', 'api'], default=None, nargs='?',
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

            for dirpath, dirnames, filenames in os.walk("presets"):
                for file in filenames:
                    file_path = os.path.join(dirpath, file)
                    print(file_path)

                    extra_args["settings_file"] = file_path

                    options["prompts"] = {
                        "0": "A solo delorean speeding on an ethereal highway through time jumps, like in the iconic movie back to the future."
                    }
                    # options["seed"] = 420
                    # options["subseed"] = 420

                    _ = deforum(**extra_args, **options)

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
    else:
        from deforum import DeforumAnimationPipeline
        deforum = DeforumAnimationPipeline.from_civitai()
        _ = deforum(**extra_args, **options)
