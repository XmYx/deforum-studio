# WARNING : This gist in the current form is a collection of command examples. Please exercise caution where mentioned.

# Docker
sudo apt-get update
sudo apt-get remove docker docker-engine docker.io
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
docker --version

# Put the user in the docker group
sudo usermod -a -G docker $USER
newgrp docker

# Nvidia Docker
sudo apt install curl
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Check Docker image
docker run --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
## Erase all Docker images [!!! CAUTION !!!]
# docker rmi -f $(docker images -a -q)

## Erase one Docker image  [!!! CAUTION !!!]
# docker ps
# docker rmi -f image_id

### Running GUI Applications
#xhost +local:docker
#docker run --gpus all -it \
#    -e DISPLAY=$DISPLAY \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    nathzi1505:darknet bash