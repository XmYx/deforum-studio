# base webui import and utils.
import json
import os
import random
import time
from datetime import date
from deforum.pipelines import DeforumAnimationPipeline

import streamlit as st
from deforum.generators.comfy_utils import ensure_comfy
from deforum.pipelines.deforum_animation.ui_params import base_params

# ensure_comfy()
# from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES
from deforum.pipelines.deforum_animation.animation_params import hybrid_params_dict
from deforum.utils.sdxl_styles import STYLE_NAMES

from deforum.utils.logging_config import logger


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


base_params = {
    "prompts": {
        "label": "Prompts",
        "type": "textbox",
        "value": "",
        "info": ""
    },
    "animation_prompts_negative": {
        "label": "Negative Prompts",
        "type": "textbox",
        "value": "",
        "info": ""
    },
    "seed": {
        "label": "Seed",
        "type": "lineedit",
        "value": "",
        "info": ""
    },


    "steps": {
        "label": "step",
        "type": "number",
        "minimum": 1,
        "maximum": 2000,
        "step": 1,
        "value": 25,
    },
    "scale": {
        "label": "scale",
        "type": "number",
        "minimum": 0.1,
        "maximum": 25.0,
        "step": 0.01,
        "value": 5.00,
    },
}

sampler_params = {
    # "sampler_name": {
    #     "label": "Sampler",
    #     "type": "dropdown",
    #     "choices": SAMPLER_NAMES,
    #     "value": SAMPLER_NAMES[0],
    # },
    # "scheduler": {
    #     "label": "Sampler",
    #     "type": "dropdown",
    #     "choices": SCHEDULER_NAMES,
    #     "value": SCHEDULER_NAMES[1],
    # },
    "width": {
        "label": "Width",
        "type": "slider",
        "minimum": 8,
        "maximum": 2048,
        "step": 8,
        "value": 768,
    },
    "height": {
        "label": "Height",
        "type": "slider",
        "minimum": 64,
        "maximum": 2048,
        "step": 8,
        "value": 768,
    },
}


base_anim_params = {
    "animation_mode": {
        "label": "Animation mode",
        "type": "radio",
        "choices": ['None', '2D', '3D', 'Video Input', 'Interpolation'],
        "value": "3D",
        "info": "control animation mode, will hide non relevant params upon change"
    },
    "max_frames": {
        "label": "Max frames",
        "type": "number",
        "step": 1,
        "value": 3,
        "minimum":1,
        "maximum":65536,
        "info": "end the animation at this frame number",
    },
    "keyframes": {
        "label": "Prompt Keys",
        "type": "textbox",
        "value": "",
        "info": ""
    },
}

class PluginInfo():
    plugname = "txt2vid"
    description = "Text to Video"
    isTab = True
    displayPriority = 1


if os.path.exists(os.path.join(st.session_state['defaults']['GFPGAN_dir'], "experiments", "pretrained_models",
                               "GFPGANv1.3.pth")):
    GFPGAN_available = True
else:
    GFPGAN_available = False

if os.path.exists(os.path.join(st.session_state['defaults']['RealESRGAN_dir'], "experiments", "pretrained_models",
                               f"{st.session_state['defaults']['RealESRGAN_model']}.pth")):
    RealESRGAN_available = True
else:
    RealESRGAN_available = False

def get_widgets(source_dict, from_dict):
    for key, value in from_dict.items():
        if value["type"] == "textbox":
            source_dict[key] = st.text_area(value['label'], value['value'], key=key)
        elif value["type"] == "lineedit":
            source_dict[key] = st.text_input(value['label'], value['value'], key=key)
        elif value["type"] == "checkbox":
            source_dict[key] = st.checkbox(value['label'], bool(value['value']), key=key)
        elif value["type"] == "radio":
            source_dict[key] = st.radio(value['label'], value['choices'], key=key, horizontal=True)
        elif value["type"] == "slider":
            source_dict[key] = st.slider(value['label'], value['minimum'], value['maximum'], value['value'], value['step'],
                                    key=key)
        elif value["type"] == "number":
            source_dict[key] = st.number_input(value['label'], value['minimum'], value['maximum'], value['value'], value['step'],
                                    key=key)
        elif value["type"] == "dropdown":
            source_dict[key] = st.selectbox(value['label'], value['choices'], key=key)
    return source_dict


def get_widgets_into_columns(source_dict, from_dict, anim_cols):
    # Sort the items based on the label or another criterion
    sorted_items = sorted(from_dict.items(), key=lambda item: item[1]['label'])

    # Distribute the sorted items across the provided columns
    for index, (key, value) in enumerate(sorted_items):
        col = anim_cols[index % len(anim_cols)]  # This will cycle through the columns
        with col:
            if value["type"] == "textbox":
                source_dict[key] = st.text_area(value['label'], value['value'], key=key)
            elif value["type"] == "lineedit":
                source_dict[key] = st.text_input(value['label'], value['value'], key=key)
            elif value["type"] == "checkbox":
                source_dict[key] = st.checkbox(value['label'], bool(value['value']), key=key)
            elif value["type"] == "radio":
                source_dict[key] = st.radio(value['label'], value['choices'], key=key, horizontal=True)
            elif value["type"] == "slider":
                source_dict[key] = st.slider(value['label'], value['minimum'], value['maximum'], value['value'], value['step'], key=key)

    return source_dict

def change_handler():
    logger.info("WOOOP")

def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(os.getcwd(), output_path, time.strftime('%Y-%m'), str(date.today().day))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    logger.info(f"Saving animation frames to {out_path}")
    os.makedirs(out_path, exist_ok=True)
    return out_path


def plugin_tab(model):
    tabs = st.tabs(["Deforum", "Area Key Editor"])
    params = {}
    with tabs[0]:
        params = main_tab(model, params)

    with tabs[1]:
        display_area_editor()


def main_tab(model=None, params=None):
    logger.info("PLUGIN_TAB")
    num_cols = 2

    with st.form("deforum"):
        cols = st.columns([3,7])

    # Creating preview object
    with cols[-1]:
        st.write()
        if 'last_image' in st.session_state:
            st.session_state.preview = st.image(st.session_state.last_image)
        else:
            st.session_state.preview = st.empty()

    # LEFT SIDE [PROMPT and SAMPLER tabs]
    with cols[0]:
        prompt_tab, sampler_tab = st.tabs(["Prompts", "Settings"])

    # PROMPT TAB
    with prompt_tab:
        params = get_widgets(params, base_params)
        submit = st.form_submit_button("Start")

    # SAMPLER TAB
    with sampler_tab:
        params = get_widgets(params, sampler_params)

    # ENABLE ANIMATION CHECKBOX
    use_animation = st.checkbox('Use Animation', on_change=change_handler)
    if use_animation:
        animation_form = st.form('animation')
        with animation_form:
            animation_tabs = st.tabs(["Anim Base", "Cadence", "Motion", "Hybrid"])
        with animation_tabs[0]:
            validate_params = st.form_submit_button('Validate Params')
        with animation_tabs[0]:
            params = get_widgets(params, base_anim_params)
        with animation_tabs[1]:
            pass
        with animation_tabs[2]:
            pass
        with animation_tabs[3]:
            num_hybrid_cols = st.slider('Columns', min_value=1, max_value=10, value=1)
            anim_cols = st.columns(num_hybrid_cols)
            params = get_widgets_into_columns(params, hybrid_params_dict, anim_cols)


    #MAIN SUBMIT LOGIC
    if submit:
        if not use_animation:
            params['max_frames'] = 1
            params['use_depth_warping'] = False
            params['keyframes'] = "0"
        prom = params['prompts']
        key = params['keyframes']
        if prom == "":
            prom = "Abstract art"
        if key == "":
            key = "0"
        new_prom = list(prom.split("\n"))
        new_key = list(key.split("\n"))
        params["animation_prompts"] = dict(zip(new_key, new_prom))
        if params['seed'] == '':
            params['seed'] = int(random.randrange(0, 4200000000))
        else:
            params['seed'] = int(params['seed'])
        # if use_settings_file and uploaded_file:
        #     for key, value in uploaded_data.items():
        #         txt2vid_copy[key] = value
        #
        # if st.session_state.use_areas == True:
        #     txt2vid_copy["use_areas"] = True
        #     txt2vid_copy["areas"] = st.session_state.areas
        def datacallback(data):
            st.session_state.last_image = data.get('image')
            if 'preview' in st.session_state:
                st.session_state.preview.image(data.get('image'))
            else:
                st.stop()


        if use_animation:
            animation = model(callback=datacallback, **params)
            if hasattr(animation, "video_path"):
                st.session_state.preview.video(animation.video_path)
            else:
                st.session_state.preview.image(animation.image)
        else:
            _ = model(callback=datacallback, **params)

    return params



#
#
def display_area_editor():
    refresh = None
    num_cols = st.slider("Set number of columns:", 1, 10, 2)
    cols = st.columns(num_cols)

    # Initialize session state
    if 'areas' not in st.session_state:
        st.session_state.areas = []

    if 'current_edit' not in st.session_state:
        st.session_state.current_edit = None

    with cols[0]:
        # Export data to a txt file
        filename = st.text_input("Filename for export:", "data.txt")
        if st.button("Export data"):
            with open(filename, 'w') as f:
                f.write(json.dumps(st.session_state.areas, indent=4))
            st.success(f"Data exported to {filename}")

        # Upload data from a txt file
        st.session_state.uploader = st.file_uploader("Choose a file to upload:", type="txt")

        logger.info(f"{st.session_state.uploader}")

        if st.session_state.uploader:
            file_content = st.session_state.uploader.read().decode()
            st.session_state.areas = json.loads(file_content)
            st.success("Data imported successfully!")
        if st.session_state.current_edit:
            area, index = st.session_state.current_edit
            keyframe_value = list(area.keys())[0]
            subprompt = area[keyframe_value][index]

            st.write(f"Editing subprompt {index + 1} of keyframe {keyframe_value}")
            subprompt["prompt"] = st.text_input("Prompt:", subprompt["prompt"], key=f'prompt_{keyframe_value}_{index}')
            subprompt["x"] = st.number_input("X:", min_value=0, max_value=4096, step=8, value=subprompt["x"], key=f'x_{keyframe_value}_{index}')
            subprompt["y"] = st.number_input("Y:", min_value=0, max_value=4096, step=8, value=subprompt["y"], key=f'y_{keyframe_value}_{index}')
            subprompt["w"] = st.number_input("W:", min_value=0, max_value=4096, step=8, value=subprompt["w"], key=f'w_{keyframe_value}_{index}')
            subprompt["h"] = st.number_input("H:", min_value=0, max_value=4096, step=8, value=subprompt["h"], key=f'h_{keyframe_value}_{index}')
            subprompt["s"] = st.number_input("S:", min_value=0.0, max_value=1.0, step=0.01, value=subprompt["s"], key=f's_{keyframe_value}_{index}')

            if st.button("Done Editing"):
                st.session_state.current_edit = None
                #st.rerun()

        else:
            # Add a new keyframe with default subprompt
            new_keyframe = st.number_input("Enter new keyframe:", value=0, step=1, key='new_keyframe')
            if st.button("Add keyframe"):
                st.session_state.areas.append({
                    str(new_keyframe): [{
                        "prompt": "",
                        "x": 0, "y": 0, "w": 1024, "h": 1024, "s": 1.0
                    }]
                })
                #st.rerun()

            # Select keyframe to add subprompt
            keyframe_options = [list(area.keys())[0] for area in st.session_state.areas]
            selected_keyframe = st.selectbox("Select keyframe to add subprompt:", keyframe_options,
                                             key='selected_keyframe')
            if st.button("Add subprompt to selected keyframe"):
                for area in st.session_state.areas:
                    if list(area.keys())[0] == selected_keyframe:
                        area[selected_keyframe].append({
                            "prompt": "",
                            "x": 0, "y": 0, "w": 1024, "h": 1024, "s": 1.0
                        })
                #st.rerun()
            if st.button('refresh'):
                st.rerun()

    display_areas_in_cols(cols[1:])

def display_areas_in_cols(columns):
    # Initialize session state
    if 'areas' not in st.session_state:
        st.session_state.areas = []

    if 'current_edit' not in st.session_state:
        st.session_state.current_edit = None

    col_idx = 0
    for area in st.session_state.areas:
        with columns[col_idx]:
            keyframe_value = list(area.keys())[0]
            st.write(f"Keyframe: {keyframe_value}")

            for index, subprompt in enumerate(area[keyframe_value]):
                st.write(subprompt)
                edit_button = st.button(f"Edit subprompt {index + 1} of keyframe {keyframe_value}")
                delete_button = st.button(f"Delete subprompt {index + 1} of keyframe {keyframe_value}")

                if edit_button:
                    st.session_state.current_edit = (area, index)
                if delete_button:
                    area[keyframe_value].pop(index)
                    #st.rerun()

            delete_keyframe = st.button(f"Delete keyframe {keyframe_value}")
            if delete_keyframe:
                st.session_state.areas.remove(area)
                #st.rerun()

        col_idx += 1
        if col_idx >= len(columns):
            col_idx = 0
# def plugin_tab(tabs, tab_names):
#     # def_runner = runner()
#     sub_tabs = st.tabs(["Deforum", "Area Schedule"])
#     with sub_tabs[0]:
#         with st.form("txt2vid-deforum"):
#             st.session_state["txt2vid"] = {}
#             params = st.session_state["txt2vid"]
#             # creating the page layout using columns
#             col1, col2, col3 = st.columns([1, 2, 1], gap="small")
#             with col1:
#                 with st.expander("Basic Settings", expanded=True):
#                     generate_button = st.form_submit_button("Generate")
#                     st.session_state["use_areas"] = st.checkbox("Use Area Schedule")
#                     update_button = st.form_submit_button("Update")
#                     params["prompt"] = st.text_area("Input Text", value='', placeholder="A corgi wearing a top hat.\nSecond Prompt")
#                     params["style"] = st.selectbox("Style", STYLE_NAMES)
#                     params["keyframes"] = st.text_area("Keyframes", "0", placeholder="0\n5\n10")
#                     params["max_frames"] = st.number_input("Max Frames:", min_value=1, max_value=2048, value=st.session_state['defaults']['max_frames'], step=1)
#                     params["W"] = st.slider("Width:", min_value=64, max_value=2048, value=st.session_state['defaults']['W'], step=64)
#                     params["H"] = st.slider("Height:", min_value=64, max_value=2048, value=st.session_state['defaults']['H'], step=64)
#                     params["scale"] = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0,
#                                                                                          max_value=30.0,
#                                                                                          value=st.session_state['defaults']['scale'],
#                                                                                          step=1e-1, format="%.1f",
#                                                                                          help="How strongly the image should follow the prompt.")
#                 sampler_tab, sequence_tab, flip_sequence_tab, frames_tab = st.tabs(["Sampler",
#                                                                                     "3D Animation Sequence",
#                                                                                     "2D Flip Sequence",
#                                                                                     "Frame Setup"
#                                                                                     ])
#                 with sampler_tab:
#                     params["steps"] = st.number_input('Sample Steps',
#                                                                            value=st.session_state[
#                                                                                'defaults']['steps'], step=1)
#                     params["sampler_name"] = st.selectbox(
#                         'Sampler',
#                         SAMPLER_NAMES,
#                         help="DDIM and PLMS are for quick results, you can use low sample steps. for the rest go up with the steps maybe start at 50 and raise from there")
#                     params["scheduler"] = st.selectbox(
#                         'Scheduler',
#                         SCHEDULER_NAMES,
#                         help="DDIM and PLMS are for quick results, you can use low sample steps. for the rest go up with the steps maybe start at 50 and raise from there")
#
#                     params["sampling_mode"] = st.selectbox(
#                         'Sampling Mode',
#                         ('bicubic', 'bilinear', 'nearest'))
#                     params["seed"] = st.text_input("Seed:",
#                                                                         value=st.session_state['defaults']['seed'],
#                                                                         help=" The seed to use, if left blank a random seed will be generated.")
#                     params["seed_behavior"] = st.selectbox(
#                         'Seed Behavior',
#                         ("iter", "fixed", "random"))
#                 with sequence_tab:
#                     # col4, col5 = st.columns([1,1], gap="medium")
#                     params["angle"] = st.text_input("Angle:", value=st.session_state[
#                         'defaults']['angle'])
#                     params["zoom"] = st.text_input("Zoom:",
#                                                                         value=st.session_state['defaults']['zoom'])
#                     params["translation_x"] = st.text_input("X Translation:",
#                                                                                  value=st.session_state[
#                                                                                      'defaults']['translation_x'])
#                     params["translation_y"] = st.text_input("Y Translation:",
#                                                                                  value=st.session_state[
#                                                                                      'defaults']['translation_y'])
#                     params["translation_z"] = st.text_input("Z Translation:",
#                                                                                  value=st.session_state[
#                                                                                      'defaults']['translation_z'])
#                     params["rotation_3d_x"] = st.text_input("X 3D Rotaion:",value=st.session_state['defaults']['rotation_3d_x'])
#                     params["rotation_3d_y"] = st.text_input("Y 3D Rotaion:",alue=st.session_state['defaults']['rotation_3d_y'])
#                     params["rotation_3d_z"] = st.text_input("Z 3D Rotaion:",value=st.session_state['defaults']['rotation_3d_z'])
#                     params["noise_schedule"] = st.text_input("Noise Schedule:",value=st.session_state['defaults']['noise_schedule'])
#                     params["strength_schedule"] = st.text_input("Strength Schedule:",value=st.session_state['defaults']['strength_schedule'])
#                     params["contrast_schedule"] = st.text_input("Contrast Schedule:",value=st.session_state[
#                                                                                          'defaults']['contrast_schedule'])
#                 with flip_sequence_tab:
#                     params["flip_2d_perspective"] = st.checkbox('Flip 2d Perspective', value=False)
#                     params["perspective_flip_theta"] = st.text_input("Flip Theta:",
#                                                                                           value=st.session_state[
#                                                                                               'defaults']['perspective_flip_theta'])
#                     params["perspective_flip_phi"] = st.text_input("Flip Phi:",
#                                                                                         value=st.session_state[
#                                                                                             'defaults']['perspective_flip_phi'])
#                     params["perspective_flip_gamma"] = st.text_input("Flip Gamma:",
#                                                                                           value=st.session_state[
#                                                                                               'defaults']['perspective_flip_gamma'])
#                     params["perspective_flip_fv"] = st.text_input("Flip FV:",
#                                                                                        value=st.session_state[
#                                                                                            'defaults']['perspective_flip_fv'])
#                 with frames_tab:
#                     basic_tab, mask_tab, init_tab = st.tabs(["Basics", "Mask", "Init Image"])
#                     with basic_tab:
#                         params["ddim_eta"] = st.number_input('DDIM ETA',
#                                                                                   value=st.session_state[
#                                                                                       'defaults']['ddim_eta'],
#                                                                                   step=1e-1, format="%.1f")
#
#                         params["make_grid"] = st.checkbox('Make Grid', value=False)
#                         params["grid_rows"] = st.number_input('Height',
#                                                                                    value=st.session_state[
#                                                                                        'defaults']['grid_rows'],
#                                                                                    step=1)
#
#                     with mask_tab:
#                         params["use_mask"] = st.checkbox('Use Mask', value=False)
#                         params["use_alpha_as_mask"] = st.checkbox('Use Alpha as Mask', value=False)
#                         params["mask_file"] = st.text_input("Init Image:",
#                                                                                  value=st.session_state[
#                                                                                      'defaults']['mask_file'],
#                                                                                  help="The Mask to be used")
#                         params["invert_mask"] = st.checkbox('Invert Mask', value=False)
#                         params["mask_brightness_adjust"] = st.number_input('Brightness Adjust',
#                                                                                                 value=st.session_state[
#                                                                                                     'defaults']['mask_brightness_adjust'],
#                                                                                                 step=1e-1,
#                                                                                                 format="%.1f",
#                                                                                                 help="Adjust the brightness of the mask")
#                         params["mask_contrast_adjust"] = st.number_input('Contrast Adjust',
#                                                                                               value=st.session_state[
#                                                                                                   'defaults']['mask_contrast_adjust'],
#                                                                                               step=1e-1, format="%.1f",
#                                                                                               help="Adjust the contrast of the mask")
#                     with init_tab:
#                         params["use_init"] = st.checkbox('Use Init', value=False)
#                         params["strength"] = st.number_input('Strength',
#                                                                                   value=st.session_state[
#                                                                                       'defaults']['strength'],
#                                                                                   step=1e-1, format="%.1f")
#                         params["strength_0_no_init"] = st.checkbox('Strength 0', value=True,
#                                                                                         help="Set the strength to 0 automatically when no init image is used")
#                         params["init_image"] = st.text_input("Init Image:",
#                                                                                   value=st.session_state[
#                                                                                       'defaults']['init_image'],
#                                                                                   help="The image to be used as init")
#             # with st.expander(""):
#             with col2:
#                 preview_tab, prompt_tab, rendering_tab, settings_tab = st.tabs(["Preview",
#                                                                                 "Hybrid Video",
#                                                                                 "Rendering",
#                                                                                 "Settings"
#                                                                                 ])
#                 with prompt_tab:
#                     params = get_widgets(params, hybrid_params_dict)
#
#
#                     params["prompt_tmp"] = st.text_area("Park your samples here", value='')
#
#                 with preview_tab:
#                     # st.write("Image")
#
#                     # create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
#
#                     st.session_state.preview_image = st.empty()
#
#                     st.session_state["loading"] = st.empty()
#
#                     st.session_state["progress_bar_text"] = st.empty()
#                     st.session_state["progress_bar"] = st.empty()
#
#                     # generate_video = st.empty()
#                     if "mp4_path" not in st.session_state:
#                         st.session_state["preview_video"] = st.empty()
#                     else:
#                         st.session_state["preview_video"] = st.video(st.session_state["mp4_path"])
#
#                     message = st.empty()
#
#                 with settings_tab:
#                     uploaded_file = st.file_uploader("Choose a settings txt file", type="txt")
#                     if uploaded_file:
#                         try:
#                             uploaded_data = json.loads(uploaded_file.getvalue())
#
#                             for key, value in uploaded_data.items():
#                                 st.session_state['defaults'][key] = value
#
#                         except Exception as e:
#                             uploaded_data = None
#                             print("Invalid settings file, please check:", e)
#                     use_settings_file = st.checkbox("Use settings file")
#                     params["save_samples"] = st.checkbox('Save Samples', value=True)
#                     params["save_settings"] = st.checkbox('Save Settings', value=False)  # For now
#                     params["display_samples"] = st.checkbox('Display Samples', value=True)
#                     params["pathmode"] = st.selectbox('Path Structure', ("subfolders", "root"),
#                                                                            index=st.session_state[
#                                                                                'defaults']['default_path_mode_index'],
#                                                                            help="subfolders structure will create daily folders plus many subfolders, root will use your outdir as root",
#                                                                            key='pathmode-txt2vid')
#                     #TODO
#                     # params["outdir"] = st.text_input("Output Folder",
#                     #                                                       value=st.session_state['defaults'].outdir,
#                     #                                                       help=" Output folder", key='outdir-txt2vid')
#                     params["filename_format"] = st.selectbox(
#                         'Filename Format',
#                         ("{timestring}_{index}_{seed}.png", "{timestring}_{index}_{prompt}.png"))
#                     params["frame_interpolation_engine"] = st.selectbox('Interpolation Engine',
#                                                                                              ("FILM", "None"),
#                                                                                              index=0,
#                                                                                              help="",
#                                                                                              key='interp-txt2vid')
#                     params["frame_interpolation_x_amount"] = st.number_input('Interpolation X Amount',
#                                                                                                   value=2,
#                                                                                                   step=1)
#                     #TODO
#                     # "frame_interpolation_slow_mo_enabled": false,
#                     # "frame_interpolation_slow_mo_amount": 2,
#                     # "frame_interpolation_keep_imgs": false,
#                     # "frame_interpolation_use_upscaled": false,
#
#             with col3:
#
#
#                 with st.expander("Animation", expanded=True):
#                     params["animation_mode"] = st.selectbox(
#                         'Animation Mode',
#                         ('None', '2D', '3D'))
#                     params["border"] = st.selectbox(
#                         'Border',
#                         ('wrap', 'replicate'))
#                     params["color_coherence"] = st.selectbox(
#                         'Color Coherence',
#                         ('Match Frame 0 LAB', 'None', 'Match Frame 0 HSV', 'Match Frame 0 RGB'))
#                     params["turbo_steps"] = st.number_input("Diffusion Cadence", min_value=0, max_value=4096, value=1, step=1)
#                     params["optical_flow_cadence"] = st.selectbox("Optical Flow Cadence", ['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback', "DenseRLOF", "SF", "DualTVL1", "DeepFlow", "PCAFlow"])
#
#
#
#                     params["use_depth_warping"] = st.checkbox('Use Depth Warping', value=True)
#                     params["midas_weight"] = st.number_input('Midas Weight',
#                                                                                   value=st.session_state[
#                                                                                       'defaults']['midas_weight'],
#                                                                                   step=1e-1,
#                                                                                   format="%.1f")
#                     params["near_plane"] = st.number_input('Near Plane',
#                                                                                 value=st.session_state[
#                                                                                     'defaults']['near_plane'],
#                                                                                 step=1)
#                     params["far_plane"] = st.number_input('Far Plane',
#                                                                                value=st.session_state[
#                                                                                    'defaults']['far_plane'],
#                                                                                step=1)
#                     params["fov"] = st.number_input('FOV',
#                                                                          value=st.session_state['defaults']['fov'],
#                                                                          step=1)
#                     params["padding_mode"] = st.selectbox(
#                         'Padding Mode',
#                         ('border', 'reflection', 'zeros'))
#
#                     params["save_depth_maps"] = st.checkbox('Save Depth Maps', value=False)
#                     params["extract_nth_frame"] = st.number_input('Extract Nth Frame',
#                                                                                        value=st.session_state[
#                                                                                            'defaults']['extract_nth_frame'],
#                                                                                        step=1)
#                     params["interpolate_key_frames"] = st.checkbox('Interpolate Key Frames',
#                                                                                         value=False)
#                     params["interpolate_x_frames"] = st.number_input('Number Frames to Interpolate',
#                                                                                           value=st.session_state[
#                                                                                               'defaults']['interpolate_x_frames'],
#                                                                                           step=1)
#                     params["resume_from_timestring"] = st.checkbox('Resume From Timestring',
#                                                                                         value=False)
#                     params["resume_timestring"] = st.text_input("Resume Timestring:",
#                                                                                      value=st.session_state[
#                                                                                          'defaults']['resume_timestring'],
#                                                                                      help="Some Video Path")
#                     params["iterations"] = 1
#                     params["batch_size"] = 1
#
#                     if update_button:
#                         st.rerun()
#                     if generate_button:
#                         # gen_args = get_args()
#
#                             # models["deforum_pipe"] = DeforumAnimationPipeline.from_file(model_path="/home/mix/Downloads/SSD-1B.safetensors")
#
#
#
#                         # frames.clear()
#
#                         txt2vid_copy = params
#                         prom = params['prompt']
#                         key = params['keyframes']
#
#                         new_prom = list(prom.split("\n"))
#                         new_key = list(key.split("\n"))
#
#                         txt2vid_copy["animation_prompts"] = dict(zip(new_key, new_prom))
#                         if txt2vid_copy['seed'] == '':
#                             txt2vid_copy['seed'] = int(random.randrange(0, 4200000000))
#                         else:
#                             txt2vid_copy['seed'] = int(
#                                 txt2vid_copy['seed'])
#
#                         if use_settings_file and uploaded_file:
#                             for key, value in uploaded_data.items():
#                                 txt2vid_copy[key] = value
#
#                         if st.session_state.use_areas == True:
#                             txt2vid_copy["use_areas"] = True
#                             txt2vid_copy["areas"] = st.session_state.areas
#                         def datacallback(data):
#                             if "preview_image" not in st.session_state:
#                                 with col2:
#                                     with preview_tab:
#                                         st.session_state["preview_image"] = st.empty()
#                             image = None
#                             if "cadence_frame" in data:
#                                 image = data['cadence_frame']
#                             elif 'image' in data:
#                                 image = data['image']
#                             if image is not None:
#                                 st.session_state["preview_image"].image(image)
#
#                         success = models["deforum_pipe"](callback=datacallback, **txt2vid_copy)
#                         if hasattr(success, "video_path"):
#                             st.session_state["preview_image"].video(success.video_path)
#     with sub_tabs[1]:
#         display_area_editor()