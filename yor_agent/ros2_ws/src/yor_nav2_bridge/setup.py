from glob import glob
import os

from setuptools import find_packages, setup


package_name = "yor_nav2_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (
            os.path.join("share", package_name, "behavior_trees"),
            glob("behavior_trees/*.xml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="YOR",
    maintainer_email="yor@example.com",
    description="ROS 2 adapters and Nav2 bringup for YOR",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "zed_bridge = yor_nav2_bridge.zed_bridge:main",
            "base_bridge = yor_nav2_bridge.base_bridge:main",
        ],
    },
)
