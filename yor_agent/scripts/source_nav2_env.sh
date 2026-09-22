#!/usr/bin/env bash
# Source this file in every shell that runs Nav2 or the YOR agent.
case "$-" in
    *u*) yor_nav2_restore_nounset=1; set +u ;;
    *) yor_nav2_restore_nounset=0 ;;
esac
source /opt/ros/humble/setup.bash
source /home/yor/codefield/YOR/yor_agent/ros2_ws/install/setup.bash

# The agent runs in capx-jetson.  Do not prepend the complete yor-nav virtual
# environment here: doing so replaces capx-jetson's NumPy 1.x with yor-nav's
# NumPy 2.x and breaks the Jetson PyTorch/Ultralytics binary extensions.
# ROS 2 and yor_nav2_bridge already publish their own Python paths through the
# two setup files above.  Also remove the old entry so re-sourcing this fixed
# script repairs an already-contaminated interactive shell.
yor_nav2_clean_pythonpath=""
IFS=: read -r -a yor_nav2_pythonpath_entries <<< "${PYTHONPATH:-}"
for yor_nav2_pythonpath_entry in "${yor_nav2_pythonpath_entries[@]}"; do
    case "${yor_nav2_pythonpath_entry}" in
        ""|/home/yor/codefield/YOR/navdp|/home/yor/venvs/capx-jetson/lib/python3.10/site-packages|/home/yor/venvs/yor-nav/lib/python3.10/site-packages)
            continue
            ;;
    esac
    yor_nav2_clean_pythonpath+="${yor_nav2_clean_pythonpath:+:}${yor_nav2_pythonpath_entry}"
done
# ROS console scripts use /usr/bin/python3, so they do not inherit the active
# capx-jetson interpreter's site-packages automatically. Append that compatible
# environment explicitly for commlink while keeping ROS/workspace packages
# ahead of it. This preserves NumPy 1.x for both the ROS bridges and the agent.
export PYTHONPATH="/home/yor/codefield/YOR/yor_agent/src:/home/yor/codefield/YOR/navdp${yor_nav2_clean_pythonpath:+:${yor_nav2_clean_pythonpath}}:/home/yor/venvs/capx-jetson/lib/python3.10/site-packages"
unset yor_nav2_clean_pythonpath
unset yor_nav2_pythonpath_entries
unset yor_nav2_pythonpath_entry
if [[ "${yor_nav2_restore_nounset}" == 1 ]]; then
    set -u
fi
unset yor_nav2_restore_nounset
