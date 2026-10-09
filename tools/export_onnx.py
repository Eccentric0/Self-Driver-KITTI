from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _project_path  # noqa: F401 - make the local src package importable

from kitti_perception.config import load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the KITTI detector to ONNX")
    parser.add_argument("--config", default="configs/kitti.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    export = config["export"]
    weights = project_path(config, args.weights or export["weights"])
    if not weights.is_file():
        raise FileNotFoundError(f"Detector weights not found: {weights}")
    output_dir = (
        args.output_dir
        if args.output_dir is not None and args.output_dir.is_absolute()
        else project_path(config, args.output_dir or export["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    generated = Path(
        model.export(
            format="onnx",
            imgsz=int(export["image_size"]),
            opset=int(export["opset"]),
            simplify=True,
            dynamic=False,
            half=False,
        )
    ).resolve()
    destination = output_dir / f"{config['train']['name']}.onnx"
    if generated != destination.resolve():
        shutil.copy2(generated, destination)
    print(json.dumps({"onnx": str(destination.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

