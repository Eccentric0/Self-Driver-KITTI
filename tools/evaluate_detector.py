from __future__ import annotations

import argparse

import _project_path  # noqa: F401 - make the local src package importable

from kitti_perception.config import load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the KITTI YOLO detector")
    parser.add_argument("--config", default="configs/kitti.yaml")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate = config["evaluate"]
    weights = project_path(config, args.weights or evaluate["weights"])
    data_yaml = project_path(config, config["data"]["dataset_yaml"])
    if not weights.is_file():
        raise FileNotFoundError(f"Detector weights not found: {weights}")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml), split="val", device=evaluate["device"], plots=True
    )
    print(f"mAP50={float(metrics.box.map50):.6f}")
    print(f"mAP50-95={float(metrics.box.map):.6f}")


if __name__ == "__main__":
    main()

