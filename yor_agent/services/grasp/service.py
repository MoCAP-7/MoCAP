"""Run the configured grasp-model service from a stable yor_agent entry point.

GraspGen-X is currently the default and only implemented backend. Its pinned
upstream package source is vendored under ``yor_agent/third_party``. This
module never imports code from a hard-coded source checkout; the checkout at
``/home/yor/codefield/GraspGenX`` is reference material only.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


DEFAULT_CHECKPOINT_ROOT = Path("/home/yor/models/GraspGenXModel/release")
DEFAULT_SERVICE_BACKEND = "graspgenx"
YOR_AGENT_ROOT = Path(__file__).resolve().parents[2]
VENDORED_GRASPGENX_ROOT = YOR_AGENT_ROOT / "third_party" / "graspgenx_upstream"
DEFAULT_PARAMETER_ASSETS = (
    YOR_AGENT_ROOT / "src" / "yor_agent" / "robot" / "assets" / "graspgenx"
)


def main(
    backend: str = DEFAULT_SERVICE_BACKEND,
    checkpoint_root: str = str(DEFAULT_CHECKPOINT_ROOT),
    assets_dir: str = str(DEFAULT_PARAMETER_ASSETS),
    host: str = "127.0.0.1",
    port: int = 5556,
    use_tensorrt: bool = False,
    tensorrt_precision: str = "fp32",
) -> None:
    """Load one shared grasp model and serve requests forever."""

    backend = str(backend).strip().lower()
    if backend != "graspgenx":
        raise ValueError(
            f"unsupported grasp service backend {backend!r}; "
            "currently available: graspgenx"
        )

    checkpoint = Path(checkpoint_root).expanduser().resolve()
    for relative in ("gen/config.yaml", "dis/config.yaml"):
        path = checkpoint / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing GraspGen-X checkpoint file: {path}")
    for subdirectory in ("gen", "dis"):
        if not any((checkpoint / subdirectory).glob("epoch_*.pth")):
            raise FileNotFoundError(
                f"missing GraspGen-X weights in {checkpoint / subdirectory}"
            )
    parameter_assets = Path(assets_dir).expanduser().resolve()
    if not parameter_assets.is_dir():
        raise FileNotFoundError(
            f"yor_agent GraspGen-X assets directory is missing: {parameter_assets}"
        )

    # The upstream package performs dependency discovery at import time.  Set
    # both locations first so an offline robot run never auto-clones into an
    # installed package or source tree. Parameter-only inference does not need
    # named gripper assets, but an existing directory suppresses that clone.
    os.environ.setdefault("GRASPGENX_CHECKPOINT_DIR", str(checkpoint.parent))
    os.environ.setdefault("GRASPGENX_GRIPPER_CFG_DIR", str(parameter_assets))
    if not (VENDORED_GRASPGENX_ROOT / "graspgenx" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"yor_agent-owned GraspGen-X source is missing: {VENDORED_GRASPGENX_ROOT}"
        )
    vendor_path = str(VENDORED_GRASPGENX_ROOT)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    try:
        from graspgenx.serving.zmq_server import GraspGenXZMQServer
    except ImportError as exc:
        raise RuntimeError(
            "The yor_agent GraspGen-X server requires its model and ZMQ "
            "dependencies in the active Python environment"
        ) from exc

    server = GraspGenXZMQServer(
        config_path=str(checkpoint),
        assets_dir=str(parameter_assets),
        host=str(host),
        port=int(port),
        default_gripper=None,
        use_tensorrt=bool(use_tensorrt),
        tensorrt_precision=str(tensorrt_precision),
    )
    server.serve_forever()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("graspgenx",),
        default=DEFAULT_SERVICE_BACKEND,
        help="grasp model served by this stable entry point",
    )
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--assets-dir", default=str(DEFAULT_PARAMETER_ASSETS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument(
        "--tensorrt-precision",
        choices=("fp32", "fp16"),
        default="fp32",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        backend=args.backend,
        checkpoint_root=args.checkpoint_root,
        assets_dir=args.assets_dir,
        host=args.host,
        port=args.port,
        use_tensorrt=args.tensorrt,
        tensorrt_precision=args.tensorrt_precision,
    )
