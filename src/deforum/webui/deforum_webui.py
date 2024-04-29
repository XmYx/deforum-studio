import ast
import os
import streamlit as st
import json

from deforum import logger

curr_folder = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(curr_folder, 'ui.json')

# Load settings from a JSON file
with open(json_path, 'r') as file:
    config = json.load(file)

# Function to send collected data to a backend function
def send_to_backend(data):
    from deforum.shared_storage import models

    if "deforum_pipe" not in models:
        load_deforum()
    st.session_state.video_path = None
    params = data.copy()
    params.pop('preview')
    use_settings_file = params.pop('use_settings_file')

    if 'settings_file' in params and use_settings_file:
        params.pop('settings_file')

        if 'settings_file' in st.session_state:

        # if st.session_state.settings_file.name is not None:
            file_path = os.path.join(curr_folder, st.session_state.settings_file)

    print(use_settings_file)

    if not use_settings_file:
        file_path = None
        params["settings_file"] = ""

        prom = params.get('prompts', 'cat sushi')
        key = params.get('keyframes', '0')
        if prom == "":
            prom = "Abstract art"
        if key == "":
            key = "0"

        if isinstance(prom, str):
            try:
                prom = ast.literal_eval(prom)
            except ValueError as e:
                pass

        if not isinstance(prom, dict):
            new_prom = list(prom.split("\n"))
            new_key = list(key.split("\n"))
            params["animation_prompts"] = dict(zip(new_key, new_prom))
        else:
            params["animation_prompts"] = prom
    def datacallback(data):
        st.session_state.last_image = data.get('image')
        if 'preview' in st.session_state:
            st.session_state.preview.image(data.get('image'))
        else:
            st.stop()

    from deforum.shared_storage import models
    animation = models["deforum_pipe"](callback=datacallback, **params) if not use_settings_file else models["deforum_pipe"](callback=datacallback, settings_file=file_path)
    if hasattr(animation, "video_path") and animation.max_frames > 1:
        st.session_state.video_path = animation.video_path
        st.session_state.preview.video(animation.video_path)
    else:
        st.session_state.preview.image(animation.image)


def update_ui_elements(settings_data):
    for key, value in settings_data.items():
        if key in st.session_state:
            print(f"Updating widget '{key}' with value: {value}")
            if isinstance(value, dict):
                value = str(value)
            st.session_state[key] = value
        else:
            print(f"No widget found for key: {key}")

def load_deforum():
    from deforum.shared_storage import models

    if "deforum_pipe" not in models:
        logger.info("LOADING DEFORUM INTO ST")
        from deforum import DeforumAnimationPipeline
        models["deforum_pipe"] = DeforumAnimationPipeline.from_civitai(model_id="125703")#, generator_name='DeforumDiffusersGenerator')
        # models["deforum_pipe"] = DeforumAnimationPipeline.from_file("/home/mix/Downloads/D4ll34_001CKPT.safetensors")
        st.session_state['loaded'] = True
    else:
        st.session_state['loaded'] = True

def main():
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    st.set_page_config(layout="wide")

    # Initialize the sidebar with tabs for each category
    tab_keys = list(config.keys())
    col1, col2 = st.columns([3, 7])

    with col1:
        tabs = st.tabs(tab_keys)

        st.button("Submit", on_click=lambda: send_to_backend({key: value for key, value in st.session_state.items()}))
        # if st.button("Submit"):
        #     send_to_backend({key: widget for key, widget in st.session_state.items()})
        for tab_key, settings in config.items():
            with tabs[tab_keys.index(tab_key)]:
                for setting, params in settings.items():
                    if params['widget_type'] == 'number':
                        st.number_input(params['label'], min_value=params['min'],
                                                           max_value=params['max'], value=params['default'], key=setting)
                    elif params['widget_type'] == 'checkbox':
                        st.checkbox(params['label'], params['default'], key=setting)
                    elif params['widget_type'] == 'dropdown':
                        st.selectbox(params['label'], params['options'], index=params['options'].index(params['default']), key=setting)
                    elif params['widget_type'] == 'text input':
                        st.text_input(params['label'], value=params['default'], key=setting)
                    elif params['widget_type'] == 'text box':
                        value = str(st.session_state.get(setting, params['default'] if 'default' in params else ''))
                        print("ERROR AT", params['label'], value, setting)

                        st.text_area(params['label'], value=str(value), key=setting)
                    elif params['widget_type'] == 'slider':
                        st.slider(params['label'], min_value=params['min'],
                                                     max_value=params['max'], value=params['default'], key=setting)
                    elif params['widget_type'] == 'file_input':
                        # Handle file input with specified accepted file types
                        file_uploader = st.file_uploader(params['label'],
                                                         type=params['accepted_file_types'],
                                                         key=st.session_state["file_uploader_key"])
                        if file_uploader is not None:
                            file_path = os.path.join(curr_folder, file_uploader.name)
                            st.session_state.settings_file = file_path

                            with open(file_path, "wb") as f:
                                f.write(file_uploader.getbuffer())
                            try:
                                # Try to open the uploaded file as JSON and update UI elements
                                with open(file_path, 'r') as settings_file:
                                    settings_data = json.load(settings_file)
                                    update_ui_elements(settings_data)
                            except json.JSONDecodeError:
                                st.error("Uploaded file is not a valid JSON.")
                            st.session_state["file_uploader_key"] += 1
                            st.rerun()



    with col2:
        if 'last_image' not in st.session_state:
            st.session_state.preview = st.empty()

main()
