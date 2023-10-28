import argparse
import os
import sys



def start_deforum_cli():
    process = None
    #model_storage.img_gen = ComfyDeforumGenerator()

    parser = argparse.ArgumentParser(description="Load settings from a txt file and run the deforum process.")

    parser.add_argument("--webui", action="store_true", help="Path to the txt file containing dictionaries to merge.")


    parser.add_argument("--file", type=str, help="Path to the txt file containing dictionaries to merge.")

    parser.add_argument("--options", nargs=argparse.REMAINDER,
                        help="Additional keyword arguments to pass to the reforum function.")
    args_main = parser.parse_args()



    extra_args = {}

    if args_main.file:
        extra_args["settings_file"] = args_main.file

    def convert_value(value):
        """Converts the string value to its corresponding data type."""
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            if value.startswith('"') and value.endswith('"'):
                return value[1:-1]  # Remove the quotes and return the string
            else:
                return value

    options = {}
    if args_main.options:
        for item in args_main.options:
            print(item)
            key, value_str = item.split('=')
            value = convert_value(value_str)
            options[key] = value

    if not args_main.webui:
        from deforum import DeforumAnimationPipeline
        deforum = DeforumAnimationPipeline.from_civitai()
        animation = deforum(**extra_args, **options)
    elif args_main.webui:
        print("start")
        import streamlit.web.cli as stcli
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stcli.main(["run", f"{root_path}/webui/deforum_webui.py"])

    #
    #
    # #deforum.enable_internal_controlnet()
    # try:
    #     if args_main.pipeline == "deforum":
    #
    #         # global img_gen
    #
    #
    #
    #
    #         deforum = setup_deforum()
    #
    #         #Switch to turn test Glow Consistency functions on or off
    #         deforum.test_flow = False
    #
    #         deforum.generate_txt2img = generate_txt2img_comfy
    #         deforum.datacallback = datacallback
    #         if args_main.file:
    #             merged_data = merge_dicts_from_txt(args_main.file)
    #             # 3. Update the SimpleNamespace objects
    #             for key, value in merged_data.items():
    #
    #                 if key == "prompts": deforum.root.animation_prompts = value
    #
    #                 if hasattr(deforum.args, key):
    #                     setattr(deforum.args, key, value)
    #                 if hasattr(deforum.anim_args, key):
    #                     setattr(deforum.anim_args, key, value)
    #                 if hasattr(deforum.parseq_args, key):
    #                     setattr(deforum.parseq_args, key, value)
    #                 if hasattr(deforum.loop_args, key):
    #                     setattr(deforum.loop_args, key, value)
    #                 if hasattr(deforum.video_args, key):
    #                     setattr(deforum.video_args, key, value)
    #         if deforum.args.seed == -1:
    #             deforum.args.seed = secrets.randbelow(18446744073709551615)
    #         with torch.inference_mode():
    #             torch.cuda.empty_cache()
    #             #sys.setrecursionlimit(10000)  # or a larger number
    #
    #             success = deforum()
    #             torch.cuda.empty_cache()
    #
    #             output_filename_base = os.path.join(deforum.args.timestring)
    #
    #             interpolator = Interpolator()
    #
    #             interpolated = interpolator(frames, 1)
    #             torch.cuda.empty_cache()
    #
    #         save_as_h264(frames, output_filename_base + ".mp4", fps=15)
    #         save_as_h264(interpolated, output_filename_base + "_FILM.mp4", fps=30)
    #         if len(cadence_frames) > 0:
    #             save_as_h264(cadence_frames, output_filename_base + f"_cadence{deforum.anim_args.diffusion_cadence}.mp4")
    #
    #     elif args_main.pipeline == "real2real":
    #         from deforum.pipelines.r2r_pipeline import Real2RealPipeline
    #         real2real = Real2RealPipeline()
    #
    #         prompts = [
    #             "Starry night, Abstract painting by picasso",
    #             "PLanets and stars on the night sky, Abstract painting by picasso",
    #             "Galaxy, Abstract painting by picasso",
    #         ]
    #
    #         keys = [30,30,30,30]
    #
    #         real2real(fixed_seed=True,
    #                   mirror_conds=False,
    #                   use_feedback_loop=True,
    #                   prompts=prompts,
    #                   keys=keys,
    #                   strength=0.45)
    #
    #     elif args_main.pipeline == "webui":
    #         import streamlit.web.cli as stcli
    #         stcli.main(["run", f"{root_path}/deforum/streamlit_ui.py"])
    #     elif args_main.pipeline == "api":
    #         import uvicorn
    #         uvicorn.run(app, host="0.0.0.0", port=8000)
    #
    #
    #     elif args_main.pipeline == "reforum":
    #         from deforum.pipelines.deforum_pipeline import DeforumAnimationPipeline
    #
    #         import deforum.datafunctions.comfy_functions
    #
    #         if args_main.lcm:
    #             lcm = True
    #         else:
    #             lcm = False
    #
    #         if args_main.trt:
    #             import deforum.datafunctions.enable_comfy_trt
    #             trt = True
    #             lcm = False
    #         else:
    #             trt = False
    #
    #
    #         deforum = DeforumAnimationPipeline.from_civitai(lcm=lcm, trt=trt)
    #         deforum.datacallback = datacallback
    #
    #         extra_args = {}
    #
    #         if args_main.file:
    #             extra_args["settings_file"] = args_main.file
    #
    #         def convert_value(value):
    #             """Converts the string value to its corresponding data type."""
    #             if value.isdigit():
    #                 return int(value)
    #             try:
    #                 return float(value)
    #             except ValueError:
    #                 if value.startswith('"') and value.endswith('"'):
    #                     return value[1:-1]  # Remove the quotes and return the string
    #                 else:
    #                     return value
    #
    #         options = {}
    #         if args_main.options:
    #             for item in args_main.options:
    #                 key, value_str = item.split('=')
    #                 value = convert_value(value_str)
    #                 options[key] = value
    #
    #         animation = deforum(store_frames_in_ram=False, **extra_args, **options)
    #
    #         # dir_path = os.path.join(root_path, 'output/video')
    #         # os.makedirs(dir_path, exist_ok=True)
    #         # output_filename_base = os.path.join(dir_path, deforum.gen.timestring)
    #         # interpolator = Interpolator()
    #         # interpolated = interpolator(frames, 1)
    #         # save_as_h264(interpolated, output_filename_base + "_FILM.mp4", fps=30)
    #
    # except KeyboardInterrupt:
    #     if process:  # Check if there's a process reference
    #         process.terminate()  # Terminate the process
    #     print("\nKeyboardInterrupt detected. Exiting...")
    #     try:
    #         interpolator = Interpolator()
    #         interpolated = interpolator(frames, 1)
    #         output_filename_base = os.path.join(deforum.args.timestring)
    #         save_as_h264(frames, output_filename_base + ".mp4", fps=15)
    #         save_as_h264(interpolated, output_filename_base + "_FILM.mp4", fps=30)
    #         if len(cadence_frames) > 0:
    #             save_as_h264(cadence_frames, output_filename_base + f"_cadence{deforum.anim_args.diffusion_cadence}.mp4")
    #     except Exception as e:
    #         print(e)
