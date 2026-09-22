"""Serve GroundingDINO the way ApexNav's upstream detector server does.

VLFM's server keeps only detections whose phrase matches a class parsed from a
VLFM-style caption ("chair . person ."). ApexNav writes captions as
"label.  ", which never parse into a matching class, so that server returns no
detections for any target; it also ignores the thresholds the client sends.
This server keeps every detection above the client's box and text thresholds,
as ApexNav's ``vlm/detector/grounding_dino.py`` does, and leaves the phrase
matching to the client.
"""

from __future__ import annotations

import argparse

from baselines.nav.vlfm.model_server import (
    DEFAULT_MODELS,
    DEFAULT_UPSTREAM,
    _prepare_imports,
    _require_cuda,
    _require_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=12181)
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    parser.add_argument(
        "--config",
        default=str(
            DEFAULT_MODELS / "src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        ),
    )
    parser.add_argument(
        "--weights", default=str(DEFAULT_MODELS / "groundingdino/groundingdino_swint_ogc.pth")
    )
    args = parser.parse_args(argv)

    _require_cuda()
    _prepare_imports(args.upstream)

    import torch
    import torchvision.transforms.functional as F
    from groundingdino.util.inference import predict
    from vlfm.vlm.detections import ObjectDetections
    from vlfm.vlm.grounding_dino import GroundingDINO
    from vlfm.vlm.server_wrapper import ServerMixin, host_model, str_to_image

    class ApexNavGroundingDINOServer(ServerMixin, GroundingDINO):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            image_tensor = F.normalize(
                F.to_tensor(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            with torch.inference_mode():
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image_tensor,
                    caption=payload["caption"],
                    box_threshold=float(payload["box_threshold"]),
                    text_threshold=float(payload["text_threshold"]),
                )
            return ObjectDetections(boxes, logits, phrases, image_source=image).to_json()

    model = ApexNavGroundingDINOServer(
        config_path=_require_file(args.config, "GroundingDINO config"),
        weights_path=_require_file(args.weights, "GroundingDINO checkpoint"),
    )
    host_model(model, name="gdino", port=args.port)
    raise RuntimeError("GroundingDINO server exited unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
