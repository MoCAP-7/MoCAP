#!/usr/bin/env bash
set -Eeuo pipefail

source /home/yor/codefield/YOR/yor_agent/scripts/source_nav2_env.sh

yor_nav2_primitive_config="${YOR_PRIMITIVE_CONFIG:-/home/yor/codefield/YOR/yor_agent/configs/primitive_config.yaml}"
yor_nav2_robot_config="${YOR_ROBOT_CONFIG:-/home/yor/codefield/YOR/yor_agent/configs/mobile_manipulation.yaml}"
yor_nav2_template="/home/yor/codefield/YOR/yor_agent/ros2_ws/src/yor_nav2_bridge/config/nav2_params.yaml"
yor_nav2_runtime_params="$(mktemp /tmp/yor-nav2-params.XXXXXX.yaml)"
trap 'rm -f -- "${yor_nav2_runtime_params}"' EXIT

PYTHONPATH="/home/yor/codefield/YOR/yor_agent/src:${PYTHONPATH:-}" \
    /home/yor/venvs/capx-jetson/bin/python -m yor_agent.nav2_params \
    --primitive-config "${yor_nav2_primitive_config}" \
    --robot-config "${yor_nav2_robot_config}" \
    --template "${yor_nav2_template}" \
    --output "${yor_nav2_runtime_params}"

ros2 launch yor_nav2_bridge yor_nav2.launch.py \
    params_file:="${yor_nav2_runtime_params}"
