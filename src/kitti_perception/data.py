from __future__ import annotations

import json
import math
import random
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from kitti_perception.linking import reference_file


KITTI_CLASSES = ("car", "truck", "pedestrian", "cyclist", "tram")
CLASS_ALIASES = {
    "Car": "car",
    "Van": "car",
    "Truck": "truck",
    "Pedestrian": "pedestrian",
    "Person_sitting": "pedestrian",
    "Cyclist": "cyclist",
    "Tram": "tram",
}


def index_by_stem(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    indexed: dict[str, Path] = {}
    for path in directory.rglob(f"*{suffix}"):
        if not path.is_file():
            continue
        if path.stem in indexed:
            raise RuntimeError(f"Duplicate stem {path.stem}: {indexed[path.stem]} and {path}")
        indexed[path.stem] = path
    return indexed


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid PNG dimensions: {path}")
    return width, height


def convert_label(source: Path, destination: Path, width: int, height: int) -> Counter[str]:
    class_to_id = {name: index for index, name in enumerate(KITTI_CLASSES)}
    lines: list[str] = []
    counts: Counter[str] = Counter()
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) < 15:
            raise ValueError(f"Malformed KITTI label: {source}:{line_number}")
        mapped = CLASS_ALIASES.get(fields[0])
        if mapped is None:  # Ignore Misc and DontCare.
            continue
        try:
            x1, y1, x2, y2 = map(float, fields[4:8])
        except ValueError as error:
            raise ValueError(f"Invalid bbox: {source}:{line_number}") from error
        x1 = min(max(x1, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        x2 = min(max(x2, 0.0), float(width))
        y2 = min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(
            f"{class_to_id[mapped]} {(x1 + x2) / (2 * width):.6f} "
            f"{(y1 + y2) / (2 * height):.6f} {(x2 - x1) / width:.6f} "
            f"{(y2 - y1) / height:.6f}"
        )
        counts[mapped] += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return counts


def select_split(
    stems: list[str], fraction: float, train_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    count = math.floor(len(stems) * fraction)
    if count < 2:
        raise ValueError("Selected subset must contain at least two images")
    rng = random.Random(seed)
    selected = rng.sample(sorted(stems), count)
    rng.shuffle(selected)
    train_count = min(count - 1, max(1, math.floor(count * train_ratio)))
    return sorted(selected[:train_count]), sorted(selected[train_count:])


def prepare_dataset(
    raw_root: Path,
    output_root: Path,
    fraction: float = 0.10,
    train_ratio: float = 0.80,
    seed: int = 42,
    dataset_yaml_path: Path | None = None,
) -> dict[str, Any]:
    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    images = index_by_stem(raw_root / "training" / "image_2", ".png")
    labels = index_by_stem(raw_root / "training" / "label_2", ".txt")
    if set(images) != set(labels):
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        raise RuntimeError(
            f"Images and labels are not aligned. Missing labels={missing_labels[:10]}, "
            f"missing images={missing_images[:10]}"
        )
    if not images:
        raise RuntimeError(f"No KITTI images found under {raw_root}")

    identity = {
        "raw_root": str(raw_root),
        "fraction": fraction,
        "train_ratio": train_ratio,
        "seed": seed,
    }
    manifest_path = output_root / "subset_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: previous.get(key) for key in identity} != identity:
            raise RuntimeError(
                "Output contains a subset made with different settings. "
                "Use another output directory or remove that subset explicitly."
            )

    train_stems, val_stems = select_split(list(images), fraction, train_ratio, seed)
    report: dict[str, Any] = dict(identity)
    report["source_images"] = len(images)
    all_counts: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    for split, stems in (("train", train_stems), ("val", val_stems)):
        split_counts: Counter[str] = Counter()
        for stem in stems:
            image = images[stem]
            strategies[reference_file(output_root / "images" / split / image.name, image)] += 1
            width, height = read_png_size(image)
            split_counts.update(
                convert_label(
                    labels[stem], output_root / "labels" / split / f"{stem}.txt", width, height
                )
            )
        all_counts.update(split_counts)
        report[split] = {
            "images": len(stems),
            "objects_by_class": dict(split_counts),
            "stems": stems,
        }

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_yaml_path = (
        dataset_yaml_path
        if dataset_yaml_path is not None
        else output_root / (
            "kitti_10pct.yaml" if math.isclose(fraction, 0.10) else "kitti.yaml"
        )
    ).expanduser().resolve()
    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(KITTI_CLASSES)},
    }
    dataset_yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    report["objects_by_class"] = dict(all_counts)
    report["link_strategies"] = dict(strategies)
    report["dataset_yaml"] = str(dataset_yaml_path)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def prepare_subset_from_prepared(
    source_root: Path,
    output_root: Path,
    fraction: float,
    train_ratio: float,
    seed: int,
    dataset_yaml_path: Path,
) -> dict[str, Any]:
    """Sample a fresh train/val split from an existing YOLO KITTI dataset."""
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    dataset_yaml_path = dataset_yaml_path.expanduser().resolve()
    if source_root == output_root:
        raise ValueError("Source and output dataset directories must differ")

    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for split in ("train", "val"):
        split_images = index_by_stem(source_root / "images" / split, ".png")
        split_labels = index_by_stem(source_root / "labels" / split, ".txt")
        if set(split_images) != set(split_labels):
            raise RuntimeError(f"Source {split} images and labels are not aligned")
        duplicates = set(images) & set(split_images)
        if duplicates:
            raise RuntimeError(f"Duplicate image stems across source splits: {sorted(duplicates)[:10]}")
        images.update(split_images)
        labels.update(split_labels)
    identity = {
        "source_prepared_root": str(source_root),
        "fraction": fraction,
        "train_ratio": train_ratio,
        "seed": seed,
    }
    manifest_path = output_root / "subset_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: previous.get(key) for key in identity} != identity:
            raise RuntimeError("Output contains a subset made with different settings. Use another output directory.")

    train_stems, val_stems = select_split(list(images), fraction, train_ratio, seed)
    report: dict[str, Any] = {**identity, "source_images": len(images)}
    strategies: Counter[str] = Counter()
    for split, stems in (("train", train_stems), ("val", val_stems)):
        for stem in stems:
            strategies[reference_file(output_root / "images" / split / images[stem].name, images[stem])] += 1
            strategies[reference_file(output_root / "labels" / split / labels[stem].name, labels[stem])] += 1
        report[split] = {"images": len(stems), "stems": stems}

    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root),
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(KITTI_CLASSES)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report["dataset_yaml"] = str(dataset_yaml_path)
    report["link_strategies"] = dict(strategies)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

