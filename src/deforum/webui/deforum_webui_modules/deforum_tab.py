# base webui import and utils.
import json
import random
import time
from datetime import date
import streamlit as st
import os

from deforum.pipelines.deforum_animation.animation_params import hybrid_params_dict
from deforum.utils.sdxl_styles import STYLE_NAMES

curr_folder = os.path.dirname(os.path.abspath(__file__))

plugin_info = {"name": "Deforum"}

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at
    # start.
    from transformers import logging

    logging.set_verbosity_error()
except:
    pass
from omegaconf import OmegaConf

st.session_state["defaults"] = OmegaConf.load(os.path.join(curr_folder, "deforum_tab.yaml"))


class PluginInfo():
    plugname = "txt2vid"
    description = "Text to Video"
    isTab = True
    displayPriority = 1


if os.path.exists(os.path.join(st.session_state['defaults'].general.GFPGAN_dir, "experiments", "pretrained_models",
                               "GFPGANv1.3.pth")):
    GFPGAN_available = True
else:
    GFPGAN_available = False

if os.path.exists(os.path.join(st.session_state['defaults'].general.RealESRGAN_dir, "experiments", "pretrained_models",
                               f"{st.session_state['defaults'].txt2vid.RealESRGAN_model}.pth")):
    RealESRGAN_available = True
else:
    RealESRGAN_available = False

def get_widgets(source_dict, from_dict):
    for key, value in from_dict.items():
        if value["type"] == "textbox":
            source_dict[key] = st.text_area(value['label'], value['value'], key=key)
        elif value["type"] == "lineedit":
            source_dict[key] = st.text_input(value['label'], bool(value['value']), key=key)
        elif value["type"] == "checkbox":
            source_dict[key] = st.checkbox(value['label'], bool(value['value']), key=key)
        elif value["type"] == "radio":
            source_dict[key] = st.radio(value['label'], value['choices'], key=key, horizontal=True)
        elif value["type"] == "slider":
            source_dict[key] = st.slider(value['label'], value['minimum'], value['maximum'], value['value'], value['step'],
                                    key=key)
    return source_dict
def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(os.getcwd(), output_path, time.strftime('%Y-%m'), str(date.today().day))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    print(f"Saving animation frames to {out_path}")
    os.makedirs(out_path, exist_ok=True)
    return out_path


def datacallback(data):
    image = None
    if "cadence_frame" in data:
        image = data['cadence_frame']
    elif 'image' in data:
        image = data['image']
    if image is not None:
        st.session_state["preview_image"].image(image)


def plugin_tab(tabs, tab_names):
    # def_runner = runner()

    with st.form("txt2vid-deforum"):
        st.session_state["txt2vid"] = {}
        params = st.session_state["txt2vid"]
        # creating the page layout using columns
        col1, col2, col3 = st.columns([1, 2, 1], gap="small")
        with col1:
            with st.expander("Basic Settings", expanded=True):
                generate_button = st.form_submit_button("Generate")
                update_button = st.form_submit_button("Update")

                params["prompt"] = st.text_area("Input Text", value='', placeholder="A corgi wearing a top hat.\nSecond Prompt")
                params["style"] = st.selectbox("Style", STYLE_NAMES)
                params["keyframes"] = st.text_area("Keyframes", "0", placeholder="0\n5\n10")

                params["max_frames"] = st.slider("Max Frames:", min_value=1, max_value=2048, value=st.session_state['defaults'].txt2vid.max_frames, step=1)

                params["W"] = st.slider("Width:", min_value=64, max_value=2048, value=st.session_state['defaults'].txt2vid.W, step=64)
                params["H"] = st.slider("Height:", min_value=64, max_value=2048, value=st.session_state['defaults'].txt2vid.H, step=64)
                params["scale"] = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0,
                                                                                     max_value=30.0,
                                                                                     value=st.session_state['defaults'].txt2vid.scale,
                                                                                     step=1e-1, format="%.1f",
                                                                                     help="How strongly the image should follow the prompt.")

        # with st.expander(""):
        with col2:
            preview_tab, prompt_tab, rendering_tab, settings_tab = st.tabs(["Preview",
                                                                            "Hybrid Video",
                                                                            "Rendering",
                                                                            "Settings"
                                                                            ])
            with prompt_tab:
                params = get_widgets(params, hybrid_params_dict)


                params["prompt_tmp"] = st.text_area("Park your samples here", value='')

            with preview_tab:
                # st.write("Image")

                # create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
                st.session_state["preview_image"] = st.empty()

                params["loading"] = st.empty()

                params["progress_bar_text"] = st.empty()
                params["progress_bar"] = st.empty()

                # generate_video = st.empty()
                if "mp4_path" not in st.session_state:
                    params["preview_video"] = st.empty()
                else:
                    params["preview_video"] = st.video(st.session_state["mp4_path"])

                message = st.empty()
            with rendering_tab:
                sampler_tab, sequence_tab, flip_sequence_tab, frames_tab = st.tabs(["Sampler",
                                                                                    "3D Animation Sequence",
                                                                                    "2D Flip Sequence",
                                                                                    "Frame Setup"
                                                                                    ])

                with sampler_tab:
                    params["steps"] = st.number_input('Sample Steps',
                                                                           value=st.session_state[
                                                                               'defaults'].txt2vid.steps, step=1)
                    params["sampler"] = st.selectbox(
                        'Sampler',
                        ("ddim", "plms", "klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"),
                        help="DDIM and PLMS are for quick results, you can use low sample steps. for the rest go up with the steps maybe start at 50 and raise from there")

                    params["sampling_mode"] = st.selectbox(
                        'Sampling Mode',
                        ('bicubic', 'bilinear', 'nearest'))
                    params["seed"] = st.text_input("Seed:",
                                                                        value=st.session_state['defaults'].txt2vid.seed,
                                                                        help=" The seed to use, if left blank a random seed will be generated.")
                    params["seed_behavior"] = st.selectbox(
                        'Seed Behavior',
                        ("iter", "fixed", "random"))
                with sequence_tab:
                    # col4, col5 = st.columns([1,1], gap="medium")
                    params["angle"] = st.text_input("Angle:", value=st.session_state[
                        'defaults'].txt2vid.angle)
                    params["zoom"] = st.text_input("Zoom:",
                                                                        value=st.session_state['defaults'].txt2vid.zoom)
                    params["translation_x"] = st.text_input("X Translation:",
                                                                                 value=st.session_state[
                                                                                     'defaults'].txt2vid.translation_x)
                    params["translation_y"] = st.text_input("Y Translation:",
                                                                                 value=st.session_state[
                                                                                     'defaults'].txt2vid.translation_y)
                    params["translation_z"] = st.text_input("Z Translation:",
                                                                                 value=st.session_state[
                                                                                     'defaults'].txt2vid.translation_z)
                    params["rotation_3d_x"] = st.text_input("X 3D Rotaion:",
                                                                                 value=st.session_state[
                                                                                     'defaults'].txt2vid.rotation_3d_x)
                    params["rotation_3d_y"] = st.text_input("Y 3D Rotaion:",
                                                                                 value=st.session_state[
                                                                                     'defaults'].txt2vid.rotation_3d_y)
                    params["rotation_3d_z"] = st.text_input("Z 3D Rotaion:",
                                                                                 value=st.session_state[
                                                                                     'defaults'].txt2vid.rotation_3d_z)
                    params["noise_schedule"] = st.text_input("Noise Schedule:",
                                                                                  value=st.session_state[
                                                                                      'defaults'].txt2vid.noise_schedule)
                    params["strength_schedule"] = st.text_input("Strength Schedule:",
                                                                                     value=st.session_state[
                                                                                         'defaults'].txt2vid.strength_schedule)
                    params["contrast_schedule"] = st.text_input("Contrast Schedule:",
                                                                                     value=st.session_state[
                                                                                         'defaults'].txt2vid.contrast_schedule)
                with flip_sequence_tab:
                    params["flip_2d_perspective"] = st.checkbox('Flip 2d Perspective', value=False)
                    params["perspective_flip_theta"] = st.text_input("Flip Theta:",
                                                                                          value=st.session_state[
                                                                                              'defaults'].txt2vid.perspective_flip_theta)
                    params["perspective_flip_phi"] = st.text_input("Flip Phi:",
                                                                                        value=st.session_state[
                                                                                            'defaults'].txt2vid.perspective_flip_phi)
                    params["perspective_flip_gamma"] = st.text_input("Flip Gamma:",
                                                                                          value=st.session_state[
                                                                                              'defaults'].txt2vid.perspective_flip_gamma)
                    params["perspective_flip_fv"] = st.text_input("Flip FV:",
                                                                                       value=st.session_state[
                                                                                           'defaults'].txt2vid.perspective_flip_fv)
                with frames_tab:
                    basic_tab, mask_tab, init_tab = st.tabs(["Basics", "Mask", "Init Image"])
                    with basic_tab:
                        params["ddim_eta"] = st.number_input('DDIM ETA',
                                                                                  value=st.session_state[
                                                                                      'defaults'].txt2vid.ddim_eta,
                                                                                  step=1e-1, format="%.1f")

                        params["make_grid"] = st.checkbox('Make Grid', value=False)
                        params["grid_rows"] = st.number_input('Height',
                                                                                   value=st.session_state[
                                                                                       'defaults'].txt2vid.grid_rows,
                                                                                   step=1)

                    with mask_tab:
                        params["use_mask"] = st.checkbox('Use Mask', value=False)
                        params["use_alpha_as_mask"] = st.checkbox('Use Alpha as Mask', value=False)
                        params["mask_file"] = st.text_input("Init Image:",
                                                                                 value=st.session_state[
                                                                                     'defaults'].txt2vid.mask_file,
                                                                                 help="The Mask to be used")
                        params["invert_mask"] = st.checkbox('Invert Mask', value=False)
                        params["mask_brightness_adjust"] = st.number_input('Brightness Adjust',
                                                                                                value=st.session_state[
                                                                                                    'defaults'].txt2vid.mask_brightness_adjust,
                                                                                                step=1e-1,
                                                                                                format="%.1f",
                                                                                                help="Adjust the brightness of the mask")
                        params["mask_contrast_adjust"] = st.number_input('Contrast Adjust',
                                                                                              value=st.session_state[
                                                                                                  'defaults'].txt2vid.mask_contrast_adjust,
                                                                                              step=1e-1, format="%.1f",
                                                                                              help="Adjust the contrast of the mask")
                    with init_tab:
                        params["use_init"] = st.checkbox('Use Init', value=False)
                        params["strength"] = st.number_input('Strength',
                                                                                  value=st.session_state[
                                                                                      'defaults'].txt2vid.strength,
                                                                                  step=1e-1, format="%.1f")
                        params["strength_0_no_init"] = st.checkbox('Strength 0', value=True,
                                                                                        help="Set the strength to 0 automatically when no init image is used")
                        params["init_image"] = st.text_input("Init Image:",
                                                                                  value=st.session_state[
                                                                                      'defaults'].txt2vid.init_image,
                                                                                  help="The image to be used as init")
            with settings_tab:
                uploaded_file = st.file_uploader("Choose a settings txt file", type="txt")
                if uploaded_file:
                    try:
                        uploaded_data = json.loads(uploaded_file.getvalue())
                    except Exception as e:
                        uploaded_data = None
                        print("Invalid settings file, please check:", e)
                use_settings_file = st.checkbox("Use settings file")
                params["save_samples"] = st.checkbox('Save Samples', value=True)
                params["save_settings"] = st.checkbox('Save Settings', value=False)  # For now
                params["display_samples"] = st.checkbox('Display Samples', value=True)
                params["pathmode"] = st.selectbox('Path Structure', ("subfolders", "root"),
                                                                       index=st.session_state[
                                                                           'defaults'].general.default_path_mode_index,
                                                                       help="subfolders structure will create daily folders plus many subfolders, root will use your outdir as root",
                                                                       key='pathmode-txt2vid')
                #TODO
                # params["outdir"] = st.text_input("Output Folder",
                #                                                       value=st.session_state['defaults'].general.outdir,
                #                                                       help=" Output folder", key='outdir-txt2vid')
                params["filename_format"] = st.selectbox(
                    'Filename Format',
                    ("{timestring}_{index}_{seed}.png", "{timestring}_{index}_{prompt}.png"))
                params["frame_interpolation_engine"] = st.selectbox('Interpolation Engine',
                                                                                         ("FILM", "None"),
                                                                                         index=0,
                                                                                         help="",
                                                                                         key='interp-txt2vid')
                params["frame_interpolation_x_amount"] = st.number_input('Interpolation X Amount',
                                                                                              value=2,
                                                                                              step=1)
                #TODO
                # "frame_interpolation_slow_mo_enabled": false,
                # "frame_interpolation_slow_mo_amount": 2,
                # "frame_interpolation_keep_imgs": false,
                # "frame_interpolation_use_upscaled": false,

        with col3:


            with st.expander("Animation", expanded=True):
                params["animation_mode"] = st.selectbox(
                    'Animation Mode',
                    ('None', '2D', '3D'))
                params["border"] = st.selectbox(
                    'Border',
                    ('wrap', 'replicate'))
                params["color_coherence"] = st.selectbox(
                    'Color Coherence',
                    ('Match Frame 0 LAB', 'None', 'Match Frame 0 HSV', 'Match Frame 0 RGB'))
                params["diffusion_cadence"] = st.selectbox(
                    'Diffusion Cadence',
                    ('1', '2', '3', '4', '5', '6', '7', '8'))
                params["use_depth_warping"] = st.checkbox('Use Depth Warping', value=True)
                params["midas_weight"] = st.number_input('Midas Weight',
                                                                              value=st.session_state[
                                                                                  'defaults'].txt2vid.midas_weight,
                                                                              step=1e-1,
                                                                              format="%.1f")
                params["near_plane"] = st.number_input('Near Plane',
                                                                            value=st.session_state[
                                                                                'defaults'].txt2vid.near_plane,
                                                                            step=1)
                params["far_plane"] = st.number_input('Far Plane',
                                                                           value=st.session_state[
                                                                               'defaults'].txt2vid.far_plane,
                                                                           step=1)
                params["fov"] = st.number_input('FOV',
                                                                     value=st.session_state['defaults'].txt2vid.fov,
                                                                     step=1)
                params["padding_mode"] = st.selectbox(
                    'Padding Mode',
                    ('border', 'reflection', 'zeros'))

                params["save_depth_maps"] = st.checkbox('Save Depth Maps', value=False)
                params["extract_nth_frame"] = st.number_input('Extract Nth Frame',
                                                                                   value=st.session_state[
                                                                                       'defaults'].txt2vid.extract_nth_frame,
                                                                                   step=1)
                params["interpolate_key_frames"] = st.checkbox('Interpolate Key Frames',
                                                                                    value=False)
                params["interpolate_x_frames"] = st.number_input('Number Frames to Interpolate',
                                                                                      value=st.session_state[
                                                                                          'defaults'].txt2vid.interpolate_x_frames,
                                                                                      step=1)
                params["resume_from_timestring"] = st.checkbox('Resume From Timestring',
                                                                                    value=False)
                params["resume_timestring"] = st.text_input("Resume Timestring:",
                                                                                 value=st.session_state[
                                                                                     'defaults'].txt2vid.resume_timestring,
                                                                                 help="Some Video Path")
                params["iterations"] = 1
                params["batch_size"] = 1


                if generate_button:
                    # gen_args = get_args()
                    from deforum.shared_storage import models
                    if "deforum_pipe" not in models:
                        print("LOADING DEFORUM INTO ST")
                        from deforum.pipelines import DeforumAnimationPipeline

                        models["deforum_pipe"] = DeforumAnimationPipeline.from_civitai(trt=False)
                        models["deforum_pipe"].datacallback = datacallback
                        time.sleep(0.5)

                    # frames.clear()

                    txt2vid_copy = params
                    prom = params['prompt']
                    key = params['keyframes']

                    new_prom = list(prom.split("\n"))
                    new_key = list(key.split("\n"))

                    txt2vid_copy["animation_prompts"] = dict(zip(new_key, new_prom))
                    if txt2vid_copy['seed'] == '':
                        txt2vid_copy['seed'] = int(random.randrange(0, 4200000000))
                    else:
                        txt2vid_copy['seed'] = int(
                            txt2vid_copy['seed'])

                    if use_settings_file and uploaded_file:
                        for key, value in uploaded_data.items():
                            txt2vid_copy[key] = value

                            try:
                                params[key].value(value)
                            except:
                                pass

                    success = models["deforum_pipe"](**txt2vid_copy)

                    if hasattr(success, "video_path"):
                        st.session_state["preview_image"].video(success.video_path)
