#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(. /etc/os-release && echo "${VERSION_ID}")" != "22.04" ]]; then
    echo "This installer is intentionally limited to Ubuntu 22.04 (JetPack 6)." >&2
    exit 2
fi
if [[ "$(dpkg --print-architecture)" != "arm64" ]]; then
    echo "This installer is intended for the Jetson arm64 host." >&2
    exit 2
fi

sudo apt-get update
sudo apt-get install -y curl software-properties-common
sudo add-apt-repository -y universe
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /tmp/yor-ros-archive-keyring.gpg
sudo install -m 0644 /tmp/yor-ros-archive-keyring.gpg \
    /usr/share/keyrings/ros-archive-keyring.gpg
ubuntu_codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME}")"
architecture="$(dpkg --print-architecture)"
repository="deb [arch=${architecture} signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${ubuntu_codename} main"
echo "${repository}" | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
sudo apt-get update
sudo apt-get install -y \
    ros-humble-ros-base \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-dev-tools \
    python3-colcon-common-extensions

# ROS Humble's generated setup files read optional variables before assigning
# defaults, which is incompatible with bash nounset. Disable it only while
# sourcing and restore this script's strict mode immediately afterwards.
set +u
source /opt/ros/humble/setup.bash
set -u
cd /home/yor/codefield/YOR/yor_agent/ros2_ws
if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init
fi
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
colcon build --symlink-install

echo
echo "ROS 2 Humble, Nav2, and yor_nav2_bridge are installed."
echo "Run: /home/yor/codefield/YOR/yor_agent/scripts/start_nav2.sh"
