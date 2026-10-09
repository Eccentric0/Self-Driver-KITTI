from __future__ import annotations

import argparse
import json

import _project_path  # noqa: F401 - make the local src package importable

from kitti_perception.config import load_config, project_path
from kitti_perception.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a KITTI-only YOLO dataset")
    parser.add_argument("--config", default="configs/kitti.yaml")
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    report = prepare_dataset(
        raw_root=project_path(config, data["raw_root"]),
        output_root=project_path(config, data["output_root"]),
        dataset_yaml_path=project_path(config, data["dataset_yaml"]),
        fraction=float(args.fraction if args.fraction is not None else data["fraction"]),
        train_ratio=float(
            args.train_ratio if args.train_ratio is not None else data["train_ratio"]
        ),
        seed=int(args.seed if args.seed is not None else config["seed"]),
    )
    summary = {
        "source_images": report["source_images"],
        "train_images": report["train"]["images"],
        "val_images": report["val"]["images"],
        "objects_by_class": report["objects_by_class"],
        "link_strategies": report["link_strategies"],
        "dataset_yaml": report["dataset_yaml"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

