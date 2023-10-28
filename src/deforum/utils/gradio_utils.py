def clean_gradio_path_strings(input_str):
    if isinstance(input_str, str) and input_str.startswith('"') and input_str.endswith('"'):
        return input_str[1:-1]
    else:
        return input_str
