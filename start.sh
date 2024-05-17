#!/bin/bash

# Function to create directory if it does not exist
create_dir_if_not_exists() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Check if .env file exists
if [ ! -f .env ]; then
    echo ".env file not found. Running install.sh..."
    ./install.sh
fi

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Menu options
options=(
    "deforum ui: PyQt6 UI for configuring and running animations"
    "deforum webui: Streamlit web UI for configuring and running animations"
    "deforum animatediff: Command-line tool for running animations"
    "deforum test: Run through all motion presets for testing purposes"
    "deforum api: FastAPI server"
    "deforum setup: Install Stable-Fast optimizations"
    "deforum runsingle --file: Run single settings file"
    "deforum config"
    "deforum unittest: Run unit test"
)

# Display menu and get user choice
echo "Select an option:"
for i in "${!options[@]}"; do
    echo "$((i+1)). ${options[i]}"
done

# Read user choice using arrow keys or number entry
read -p "Enter the number of your choice: " choice

# Execute the corresponding command
case $choice in
    1)
        deforum ui
        ;;
    2)
        deforum webui
        ;;
    3)
        deforum animatediff
        ;;
    4)
        deforum test
        ;;
    5)
        deforum api
        ;;
    6)
        deforum setup
        ;;
    7)
        # Ask for the preset file path
        read -p "Enter the preset file path (default: ~/deforum/presets/preset.txt): " preset_file
        preset_file=${preset_file:-~/deforum/presets/preset.txt}
        deforum runsingle --file "$preset_file"
        ;;
    8)
        deforum config
        ;;
    9)
        deforum unittest
        ;;
    *)
        echo "Invalid choice."
        ;;
esac
