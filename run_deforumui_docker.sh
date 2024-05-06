#!/bin/bash

# Define the absolute path to the deforum directory
DEFORUM_DIR="'$(pwd)'/storage""

# Ensure the directory exists and set permissions (if necessary)
mkdir -p "$DEFORUM_DIR"
chmod 777 "$DEFORUM_DIR"  # Use more restrictive permissions as necessary

# Check if the Docker image already exists
IMAGE_NAME="deforum-desktop"
IMAGE_EXISTS=$(docker images -q $IMAGE_NAME)

# Build the Docker image only if it does not exist
if [ -z "$IMAGE_EXISTS" ]; then
    echo "Image $IMAGE_NAME does not exist. Building..."
    docker build -f Dockerfile-desktop -t $IMAGE_NAME .
else
    echo "Image $IMAGE_NAME already exists. Skipping build."
fi

# Allow X11 forwarding
xhost +local:root

# Run the Docker container
docker run -it --rm \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$DEFORUM_DIR":/root/deforum \
    --network host \
    $IMAGE_NAME

# Revoke X11 permissions after the container stops
xhost -local:root
