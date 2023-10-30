import argparse
import os


def start_deforum_cli():

    parser = argparse.ArgumentParser(description="Load settings from a txt file and run the deforum process.")
    # Positional mode argument
    parser.add_argument("mode", choices=['webui', 'animatediff', 'runpresets'], default=None, nargs='?',
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
        if in_value.isdigit():
            return int(in_value)
        try:
            return float(in_value)
        except ValueError:
            if in_value.startswith('"') and in_value.endswith('"'):
                return in_value[1:-1]  # Remove the quotes and return the string
            else:
                return in_value

    options = {}
    if args_main.options:
        for item in args_main.options:
            print(item)
            key, value_str = item.split('=')
            value = convert_value(value_str)
            options[key] = value


    if args_main.mode:
        if args_main.mode == "webui":
            import streamlit.web.cli as stcli
            root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            stcli.main(["run", f"{root_path}/webui/deforum_webui.py"])
        elif args_main.mode == "animatediff":
            from deforum.pipelines.animatediff_animation.pipeline_animatediff_animation import DeforumAnimateDiffPipeline
            pipe = DeforumAnimateDiffPipeline.from_civitai()
            _ = pipe(**extra_args, **options)

        elif args_main.mode == "runpresets":
            from deforum import DeforumAnimationPipeline
            deforum = DeforumAnimationPipeline.from_civitai()

            filepath = "/home/mix/Documents/GitHub/deforum-studio/deforum-studio/presets/Shapes-Kalidascope.txt"
            for dirpath, dirnames, filenames in os.walk("presets"):
                for file in filenames:
                    file_path = os.path.join(dirpath, file)
                    print(file_path)

                    extra_args["settings_file"] = file_path

                    options["prompts"] = {
                        "0": "travelling towards the core of earth, highly detailed illustration"
                    }
                    options["seed"] = 420
                    options["subseed"] = 420

                    _ = deforum(**extra_args, **options)
    else:
        from deforum import DeforumAnimationPipeline
        deforum = DeforumAnimationPipeline.from_civitai()
        _ = deforum(**extra_args, **options)
