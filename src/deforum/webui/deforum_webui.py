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
    use_settings_file = False
    params = data.copy()
    params.pop('preview')
    if 'settings_file' in params:
        params.pop('settings_file')
        if st.session_state.settings_file.name is not None:
            file_path = os.path.join(curr_folder, st.session_state.settings_file.name)
            use_settings_file = True

    # use_settings = False
    # if "settings_file" in st.session_state:
    #     if file_uploader:
    #         file_path = os.path.join(curr_folder, file_uploader.name)
    #         use_settings = True
    #         print("USING", file_path)
    #     else:
    #         print("Data sent to backend:", data)
    #
    # else:
    #     print("Data sent to backend:", data)
    #
    def datacallback(data):
        st.session_state.last_image = data.get('image')
        if 'preview' in st.session_state:
            st.session_state.preview.image(data.get('image'))
        else:
            st.stop()

    from deforum.shared_storage import models
    animation = models["deforum_pipe"](callback=datacallback, **params) if not use_settings_file else models["deforum_pipe"](callback=datacallback, settings_file=file_path)
    if hasattr(animation, "video_path"):
        st.session_state.preview.video(animation.video_path)
    else:
        st.session_state.preview.image(animation.image)


def update_ui_elements(settings_data):
    for key, value in settings_data.items():
        if key in st.session_state:
            print(f"Updating widget '{key}' with value: {value}")
            st.session_state[key] = value
        else:
            print(f"No widget found for key: {key}")


def main():
    from deforum.shared_storage import models

    def load_deforum():
        if "deforum_pipe" not in models:
            logger.info("LOADING DEFORUM INTO ST")
            from deforum import DeforumAnimationPipeline
            models["deforum_pipe"] = DeforumAnimationPipeline.from_civitai(model_id="125703", generator_name='DeforumDiffusersGenerator')
            # models["deforum_pipe"] = DeforumAnimationPipeline.from_file("/home/mix/Downloads/D4ll34_001CKPT.safetensors")
        else:
            st.session_state['loaded'] = True


    if 'loaded' not in st.session_state:
        st.session_state['loaded'] = True
        load_deforum()

    # Initialize the sidebar with tabs for each category
    tab_keys = list(config.keys())
    tabs = st.sidebar.tabs(tab_keys)

    for tab_key, settings in config.items():
        with tabs[tab_keys.index(tab_key)]:
            for setting, params in settings.items():
                if params['widget_type'] == 'number':
                    st.number_input(params['label'], min_value=params['min'],
                                                       max_value=params['max'], value=params['default'], key=setting)
                elif params['widget_type'] == 'dropdown':
                    st.selectbox(params['label'], params['options'], index=params['options'].index(params['default']), key=setting)
                elif params['widget_type'] == 'text input':
                    st.text_input(params['label'], value=params['default'], key=setting)
                elif params['widget_type'] == 'slider':
                    st.slider(params['label'], min_value=params['min'],
                                                 max_value=params['max'], value=params['default'], key=setting)
                elif params['widget_type'] == 'file_input':
                    # Handle file input with specified accepted file types
                    file_uploader = st.file_uploader(params['label'], type=params['accepted_file_types'])
                    if file_uploader is not None:
                        st.session_state.settings_file = file_uploader
                        file_path = os.path.join(curr_folder, file_uploader.name)
                        with open(file_path, "wb") as f:
                            f.write(file_uploader.getbuffer())
                        try:
                            # Try to open the uploaded file as JSON and update UI elements
                            with open(file_path, 'r') as settings_file:
                                settings_data = json.load(settings_file)
                                update_ui_elements(settings_data)
                        except json.JSONDecodeError:
                            st.error("Uploaded file is not a valid JSON.")

    load_deforum()

    # Main page layout
    col1, col2 = st.columns([3, 2])

    with col1:
        if st.button("Submit"):
            send_to_backend({key: widget for key, widget in st.session_state.items()})

    with col2:
        if 'last_image' in st.session_state:
            st.session_state.preview = st.image(st.session_state.last_image)
        else:
            st.session_state.preview = st.empty()

main()
