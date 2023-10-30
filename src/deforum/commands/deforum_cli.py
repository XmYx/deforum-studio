import argparse
import os


def start_deforum_cli():

    parser = argparse.ArgumentParser(description="Load settings from a txt file and run the deforum process.")
    parser.add_argument("--webui", action="store_true", help="Path to the txt file containing dictionaries to merge.")
    parser.add_argument("--animatediff", action="store_true", help="Path to the txt file containing dictionaries to merge.")
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

    if not args_main.webui:
        if args_main.animatediff:
            from deforum.pipelines.animatediff_animation.pipeline_animatediff_animation import DeforumAnimateDiffPipeline
            # pipe = DeforumAnimateDiffPipeline.from_civitai()
            pipe = DeforumAnimateDiffPipeline.from_single_file(pretrained_model_repo_or_path="/home/mix/Downloads/SSD-1B.safetensors")
            print(pipe)
        else:

            from deforum import DeforumAnimationPipeline
            deforum = DeforumAnimationPipeline.from_civitai()
            _ = deforum(**extra_args, **options)
    elif args_main.webui:
        print("start")
        import streamlit.web.cli as stcli
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stcli.main(["run", f"{root_path}/webui/deforum_webui.py"])
