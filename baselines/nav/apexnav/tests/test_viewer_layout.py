import json
from pathlib import Path
import re
import unittest

from baselines.nav.apexnav.run import VIEWER_TOPIC_WHITELIST

BASELINE_DIR = Path(__file__).resolve().parents[1]
FIXED_LAYOUT = BASELINE_DIR / "viewer" / "apexnav_yor_lichtblick_layout.json"
FOLLOW_LAYOUT = BASELINE_DIR / "viewer" / "apexnav_yor_lichtblick_layout_follow.json"


def _exposed(topic):
    # foxglove_bridge full-matches each whitelist pattern and ignores case.
    return any(
        re.fullmatch(pattern, topic, re.IGNORECASE) for pattern in VIEWER_TOPIC_WHITELIST
    )


def _layout(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _layout_topics(layout):
    topics = set()
    for panel in layout["configById"].values():
        topics.update(panel.get("topics", {}))
        image_topic = panel.get("imageMode", {}).get("imageTopic")
        if image_topic:
            topics.add(image_topic)
        topics.update(path["value"].split(".", 1)[0] for path in panel.get("paths", []))
    return topics


class ViewerLayoutTests(unittest.TestCase):
    def test_every_layout_topic_is_served_by_the_bridge(self):
        for path in (FIXED_LAYOUT, FOLLOW_LAYOUT):
            topics = _layout_topics(_layout(path))
            self.assertIn("/grid_map/free", topics)
            self.assertIn("/apexnav/detector/detect_img/compressed", topics)
            self.assertEqual(
                sorted(topic for topic in topics if not _exposed(topic)), [], path.name
            )

    def test_large_or_commanding_topics_are_not_served(self):
        for topic in (
            "/grid_map/depth_cloud",
            "/grid_map/filtered_depth_cloud",
            "/grid_map/unknown",
            "/grid_map/esdf",
            "/apexnav/rgb",
            "/apexnav/depth_normalized",
            "/apexnav/detector/detect_image",
            "/apexnav/start",
            "/APEXNAV/START",
            "/apexnav/manual_goal",
            "/apexnav/planning/trajectory",
            "/apexnav/traj_server/stop",
            "/apexnav/solve_tsp",
            "/robot",
            "/grid_map/free/extra",
        ):
            self.assertFalse(_exposed(topic), topic)

    def test_layout_tree_places_every_panel(self):
        for path in (FIXED_LAYOUT, FOLLOW_LAYOUT):
            layout = _layout(path)
            placed = set()

            def walk(node):
                if isinstance(node, str):
                    placed.add(node)
                else:
                    walk(node["first"])
                    walk(node["second"])

            walk(layout["layout"])
            self.assertEqual(placed, set(layout["configById"]), path.name)

    def test_views_hide_the_executed_path_and_differ_in_what_they_follow(self):
        fixed = _layout(FIXED_LAYOUT)["configById"]["3D!apexnavmap"]
        follow = _layout(FOLLOW_LAYOUT)["configById"]["3D!apexnavmap"]
        self.assertEqual((fixed["followTf"], fixed["followMode"]), ("world", "follow-none"))
        self.assertEqual(
            (follow["followTf"], follow["followMode"]), ("base_footprint", "follow-position")
        )
        for view in (fixed, follow):
            self.assertFalse(view["topics"]["/travel_traj"]["visible"])
        self.assertEqual(fixed["topics"], follow["topics"])

    def test_bootstrap_script_installs_the_bridge_pinned_in_upstream_lock(self):
        lock = json.loads((BASELINE_DIR / "upstream.lock").read_text(encoding="utf-8"))
        script = (BASELINE_DIR / "scripts" / "bootstrap_viewer_deps.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(f'DEB_URL="{lock["foxglove_bridge_deb_url"]}"', script)
        self.assertIn(f"DEB_SIZE={lock['foxglove_bridge_deb_size']}", script)
        self.assertIn(f'DEB_SHA256="{lock["foxglove_bridge_deb_sha256"]}"', script)


if __name__ == "__main__":
    unittest.main()
