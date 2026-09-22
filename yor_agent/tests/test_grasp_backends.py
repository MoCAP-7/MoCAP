from __future__ import annotations

import unittest

import numpy as np

from yor_agent.robot.grasp_backends import GraspGenXBackend, create_grasp_backend


def _config(**overrides):
    config = {
        "grasp_backend": "graspgenx",
        "graspgenx_planner": "diffusion",
        "graspgenx_sweep_extents_open_m": [0.065, 0.020, 0.060],
        "graspgenx_sweep_offset_open_m": [0.0, 0.0, 0.11],
        "graspgenx_sweep_extents_mid_m": [0.0325, 0.020, 0.060],
        "graspgenx_sweep_offset_mid_m": [0.0, 0.0, 0.11],
        "graspgenx_origin_to_tcp_m": 0.13,
        "graspgenx_topk_num_grasps": 1,
    }
    config.update(overrides)
    return config


class GraspGenXBackendTest(unittest.TestCase):
    def test_factory_defaults_to_graspgenx_and_piper_profile(self) -> None:
        backend = create_grasp_backend({})

        self.assertIsInstance(backend, GraspGenXBackend)
        self.assertEqual(backend.address, "tcp://127.0.0.1:5556")
        self.assertEqual(backend.planner, "diffusion")
        self.assertEqual(
            backend.gripper_profile_source,
            "official-profile:piper_hand",
        )

    def test_official_piper_profile_matches_published_sweep_volume(self) -> None:
        backend = GraspGenXBackend(
            _config(
                graspgenx_gripper_profile="piper_hand",
                graspgenx_sweep_extents_open_m=None,
                graspgenx_sweep_offset_open_m=None,
                graspgenx_sweep_extents_mid_m=None,
                graspgenx_sweep_offset_mid_m=None,
            )
        )

        self.assertEqual(
            backend.gripper_profile_source, "official-profile:piper_hand"
        )
        np.testing.assert_allclose(
            backend.sweep_volume_params["extents_open"], [0.065, 0.020, 0.055]
        )
        np.testing.assert_allclose(
            backend.sweep_volume_params["offset_open"], [0.0, 0.0, 0.105]
        )
        np.testing.assert_allclose(
            backend.sweep_volume_params["extents_mid"], [0.033, 0.020, 0.055]
        )
        self.assertAlmostEqual(
            backend.sweep_volume_params["fingertip_depth"], 0.13
        )

    def test_factory_builds_agent_local_graspgenx_backend(self) -> None:
        backend = create_grasp_backend(_config())

        self.assertIsInstance(backend, GraspGenXBackend)
        self.assertEqual(backend.address, "tcp://127.0.0.1:5556")
        self.assertEqual(backend.planner, "diffusion")

    def test_scene_depth_protocol_topk_and_tcp_anchor(self) -> None:
        backend = GraspGenXBackend(_config())
        requests = []
        lower = np.eye(4, dtype=np.float32)
        lower[:3, 3] = [0.0, 0.0, 0.5]
        higher = np.eye(4, dtype=np.float32)
        higher[:3, 3] = [0.1, 0.0, 0.5]

        def request(payload):
            requests.append(payload)
            return {
                "instance_ids": np.asarray([1], dtype=np.int32),
                "grasps": [np.asarray([lower, higher])],
                "confidences": [np.asarray([0.2, 0.9], dtype=np.float32)],
                "branch_tags": [["diff", "diff"]],
                "timing": {"infer_ms": 12.5},
                "skipped_instance_ids": np.empty((0,), dtype=np.int32),
            }

        backend._request = request  # type: ignore[method-assign]
        depth = np.full((4, 5), 0.5, dtype=np.float32)
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 1:4] = 1

        result = backend(depth, np.eye(3), mask, 1)

        self.assertEqual(requests[0]["action"], "infer_scene_depth")
        self.assertEqual(requests[0]["planner"], "diffusion")
        np.testing.assert_array_equal(requests[0]["instance_mask"], mask)
        np.testing.assert_allclose(result.poses, [higher])
        np.testing.assert_allclose(result.scores, [0.9])
        np.testing.assert_allclose(result.anchor_points, [[0.1, 0.0, 0.63]])
        self.assertEqual(result.metadata["branch_tags"], ["diff"])
        self.assertEqual(result.metadata["server_timing"], {"infer_ms": 12.5})

    def test_rejects_invalid_sweep_volume(self) -> None:
        with self.assertRaisesRegex(ValueError, "extents must be positive"):
            GraspGenXBackend(
                _config(graspgenx_sweep_extents_open_m=[0.0, 0.02, 0.06])
            )

    def test_reports_server_skip(self) -> None:
        backend = GraspGenXBackend(_config())
        backend._request = lambda payload: {  # type: ignore[method-assign]
            "instance_ids": np.empty((0,), dtype=np.int32),
            "grasps": [],
            "confidences": [],
            "skipped_instance_ids": np.asarray([1], dtype=np.int32),
        }
        with self.assertRaisesRegex(RuntimeError, "insufficient valid points"):
            backend(
                np.ones((2, 2), dtype=np.float32),
                np.eye(3),
                np.ones((2, 2), dtype=np.uint8),
                1,
            )


if __name__ == "__main__":
    unittest.main()
