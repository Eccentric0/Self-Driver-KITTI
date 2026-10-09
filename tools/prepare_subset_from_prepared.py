from __future__ import annotations

import argparse
import json

import _project_path  # noqa: F401 - make the local src package importable

from kitti_perception.config import load_config, project_path
from kitti_perception.data import prepare_subset_from_prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample KITTI from an existing prepared YOLO dataset")
    parser.add_argument("--config", default="configs/kitti.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    report = prepare_subset_from_prepared(
        source_root=project_path(config, data["source_prepared_root"]),
        output_root=project_path(config, data["output_root"]),
        dataset_yaml_path=project_path(config, data["dataset_yaml"]),
        fraction=float(data["fraction"]),
        train_ratio=float(data["train_ratio"]),
        seed=int(config["seed"]),
    )
    print(json.dumps({
        "source_images": report["source_images"],
        "train_images": report["train"]["images"],
        "val_images": report["val"]["images"],
        "dataset_yaml": report["dataset_yaml"],
        "link_strategies": report["link_strategies"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
