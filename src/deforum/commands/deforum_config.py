import os
import json

# Define paths
home_dir = os.path.expanduser("~")
config_path = os.path.join(home_dir, "deforum", ".deforumprofile")
default_venv_path = os.path.join(home_dir, 'deforum', 'venv')
default_comfy_path = os.path.join(home_dir, 'deforum', 'ComfyUI')

# Ensure the configuration directory exists
os.makedirs(os.path.dirname(config_path), exist_ok=True)

def load_config():
    """ Load the existing config file or return an empty dict if none exists. """
    try:
        with open(config_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_config(config):
    """ Save the configuration to a file. """
    with open(config_path, "w") as file:
        json.dump(config, file, indent=4)

def get_input(prompt, current_value=None):
    """ Prompt for input, showing the current value if available. Accept empty input to retain current value. """
    user_input = input(f"{prompt} [Current: {current_value if current_value is not None else 'None'}]: ")
    return user_input.strip()

def main():
    config = load_config()
    # Parameters with their types and default values
    parameters = {
        "comfy_path": ("text", default_comfy_path),
        "venv_path": ("text", default_venv_path),
        "enable_logging": ("bool", False),
        "log_level": ("choice", ["DEBUG", "INFO", "WARNING", "ERROR"], "INFO")
    }

    for key, (type_, *args) in parameters.items():
        current_value = config.get(key, args[0] if args else None)
        if type_ == "bool":
            choice = get_input(f"Set {key} (true/false)", str(current_value))
            config[key] = choice.lower() in ["true", "1", "yes"] if choice else current_value
        elif type_ == "choice":
            options, default = args
            options_str = "/".join(options)
            choice = get_input(f"Choose {key} ({options_str})", current_value)
            config[key] = choice if choice in options and choice else current_value
        elif type_ == "text":
            choice = get_input(f"Enter {key}", current_value)
            config[key] = choice if choice else current_value

    # Optionally add a confirmation before saving
    confirm = input("Save changes? (y/n): ")
    if confirm.lower() == 'y':
        save_config(config)
        print("Configuration saved.")
    else:
        print("Changes not saved.")

if __name__ == "__main__":
    main()
