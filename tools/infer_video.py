from __future__ import annotations

import argparse
import json
from pathlib import Path

import _project_path  # noqa: F401 - make the local src package importable

from kitti_perception.config import load_config, project_path
from kitti_perception.video import run_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO + ByteTrack on a video")
    parser.add_argument("--config", default="configs/kitti.yaml")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    inference = config["inference"]
    weights = project_path(config, args.weights or inference["weights"])
    source = args.source if args.source.is_absolute() else project_path(config, args.source)
    output = args.output if args.output.is_absolute() else project_path(config, args.output)
    summary = run_video(
        weights=weights,
        source=source,
        output=output,
        device=inference["device"],
        confidence=float(inference["confidence"]),
        iou=float(inference["iou"]),
        tracker=str(inference["tracker"]),
        trail_length=int(inference["trail_length"]),
        codec=str(inference["output_codec"]),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

