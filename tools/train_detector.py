from __future__ import annotations

import argparse

import _project_path  # noqa: F401 - make the local src package importable

from kitti_perception.config import load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on the prepared KITTI subset")
    parser.add_argument("--config", default="configs/kitti.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    train = config["train"]
    data_yaml = project_path(config, config["data"]["dataset_yaml"])
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}. Run tools/prepare_dataset.py first."
        )
    from ultralytics import YOLO

    model = YOLO(args.model or train["model"])
    model.train(
        data=str(data_yaml),
        imgsz=int(train["image_size"]),
        batch=int(args.batch if args.batch is not None else train["batch_size"]),
        epochs=int(args.epochs if args.epochs is not None else train["epochs"]),
        workers=int(args.workers if args.workers is not None else train["workers"]),
        device=train["device"],
        amp=bool(train["amp"]),
        seed=int(config["seed"]),
        deterministic=True,
        project=str(project_path(config, train["project"])),
        name=str(train["name"]),
        plots=True,
    )


if __name__ == "__main__":
    main()

