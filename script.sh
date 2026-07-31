# Docker Image pull command
docker pull infyartifactory.jfrog.io/vision-ops-sdk-docker/visionops-sdk-v1:latest
# Make Repo for mounting
mkdir visionopssdk
mkdir visionopssdk/framework_config
mkdir visionopssdk/raw_training_data
mkdir visionopssdk/Logs
mkdir visionopssdk/model_registry
mkdir visionopssdk/gt_reports

# Create Volume Mounts
docker volume create --driver local \
    --opt type=none \
    --opt device=$(pwd)/visionopssdk/framework_config \
    --opt o=bind \
    framework_config
docker volume create --driver local \
    --opt type=none \
    --opt device=$(pwd)/visionopssdk/raw_training_data\
    --opt o=bind \
    raw_training_data
docker volume create --driver local \
    --opt type=none \
    --opt device=$(pwd)/visionopssdk/Logs \
    --opt o=bind \
    Logs
docker volume create --driver local \
    --opt type=none \
    --opt device=$(pwd)/visionopssdk/model_registry \
    --opt o=bind \
    model_registry
docker volume create --driver local \
    --opt type=none \
    --opt device=$(pwd)/visionopssdk/gt_reports \
    --opt o=bind \
    gt_reports

# Create Container using volume mount and Image pulled from docker Image repository
# User Input for creating docker container with use case name
read -p "Please Provide use case name " my_var
docker run -d -it --name mlops_sdk_v1.0-$my_var -v framework_config:/app/framework_config -v raw_training_data:/app/raw_training_data -v gt_reports:/app/gt_reports -v Logs:/app/Logs -v model_registry:/app/model_registry infyartifactory.jfrog.io/vision-ops-sdk-docker/visionops-sdk-v1:latest
