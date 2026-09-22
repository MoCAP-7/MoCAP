"""Serve ApexNav's original BLIP2-ITM model on CPU or CUDA.

CPU is the default for Jetson memory safety: the allocation stays swappable and
cannot evict the detector. CUDA scores each image far faster, but only fits once
no other GPU services share the unified memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.nav.vlfm.model_server import (
    DEFAULT_MODELS,
    DEFAULT_UPSTREAM,
    _install_image_only_decord_stub,
    _prepare_imports,
    _require_cuda,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=12182)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    parser.add_argument("--cache-dir", default=str(DEFAULT_MODELS / "lavis/cache"))
    args = parser.parse_args(argv)

    if args.device == "cuda":
        _require_cuda()
    _prepare_imports(args.upstream)
    _install_image_only_decord_stub()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    from lavis.common.registry import registry

    registry.mapping["paths"]["cache_root"] = str(cache_dir)

    import torch
    from vlfm.vlm.blip2itm import BLIP2ITM
    from vlfm.vlm.server_wrapper import ServerMixin, host_model, str_to_image

    class BLIP2ITMServer(ServerMixin, BLIP2ITM):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {"response": self.cosine(image, payload["txt"])}

    model = BLIP2ITMServer(device=torch.device(args.device))
    host_model(model, name="blip2itm", port=args.port)
    raise RuntimeError("BLIP-2 ITM server exited unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
