from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

import _project_path  # noqa: F401 - make the local src package importable

from kitti_perception.config import load_config, project_path


def validate_split(root: Path, split: str, class_count: int) -> dict[str, object]:
    images = {path.stem: path for path in (root / "images" / split).glob("*.png")}
    labels = {path.stem: path for path in (root / "labels" / split).glob("*.txt")}
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    errors: list[str] = []
    instances: Counter[int] = Counter()
    for stem, label_path in labels.items():
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = line.split()
            if len(fields) != 5:
                errors.append(f"{label_path}:{line_number}: expected 5 fields")
                continue
            try:
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError:
                errors.append(f"{label_path}:{line_number}: non-numeric value")
                continue
            if not 0 <= class_id < class_count:
                errors.append(f"{label_path}:{line_number}: invalid class {class_id}")
            if any(value < 0.0 or value > 1.0 for value in values):
                errors.append(f"{label_path}:{line_number}: bbox outside [0, 1]")
            if values[2] <= 0.0 or values[3] <= 0.0:
                errors.append(f"{label_path}:{line_number}: non-positive bbox")
            instances[class_id] += 1
    return {
        "images": len(images),
        "labels": len(labels),
        "instances_by_class_id": dict(instances),
        "missing_labels": missing_labels[:20],
        "missing_images": missing_images[:20],
        "errors": errors[:50],
        "valid": not missing_labels and not missing_images and not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the prepared KITTI YOLO dataset")
    parser.add_argument("--config", default="configs/kitti.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_yaml = project_path(config, config["data"]["dataset_yaml"])
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Prepared dataset YAML not found: {dataset_yaml}")
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(dataset["path"])
    class_count = len(dataset["names"])
    report = {
        split: validate_split(root, split, class_count) for split in ("train", "val")
    }
    report["valid"] = all(bool(report[split]["valid"]) for split in ("train", "val"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

