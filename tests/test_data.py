import struct
import tempfile
import unittest
from pathlib import Path

from kitti_perception.data import convert_label, prepare_dataset, select_split


def write_png_header(path: Path, width: int = 1242, height: int = 375) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
    )


class DataTests(unittest.TestCase):
    def test_convert_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "label.txt"
            source.write_text(
                "Car 0 0 0 100 50 300 150 0 0 0 0 0 0 0\n"
                "Van 0 0 0 20 10 120 60 0 0 0 0 0 0 0\n"
                "DontCare 0 0 0 0 0 10 10 0 0 0 0 0 0 0\n",
                encoding="utf-8",
            )
            output = root / "output.txt"
            counts = convert_label(source, output, 1000, 200)
            self.assertEqual(dict(counts), {"car": 2})
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

    def test_split_is_deterministic(self) -> None:
        stems = [f"{index:06d}" for index in range(100)]
        first = select_split(stems, 0.10, 0.80, 42)
        second = select_split(list(reversed(stems)), 0.10, 0.80, 42)
        self.assertEqual(first, second)
        self.assertEqual((len(first[0]), len(first[1])), (8, 2))

    def test_prepare_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "raw" / "training" / "image_2"
            label_dir = root / "raw" / "training" / "label_2"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for index in range(20):
                stem = f"{index:06d}"
                write_png_header(image_dir / f"{stem}.png")
                (label_dir / f"{stem}.txt").write_text(
                    "Car 0 0 0 100 50 300 150 0 0 0 0 0 0 0\n", encoding="utf-8"
                )
            output = root / "prepared"
            report = prepare_dataset(root / "raw", output, 0.10, 0.50, 42)
            self.assertEqual(report["train"]["images"], 1)
            self.assertEqual(report["val"]["images"], 1)
            self.assertTrue((output / "kitti_10pct.yaml").is_file())
            self.assertTrue((output / "subset_manifest.json").is_file())

            full_output = root / "kitti_100pct"
            full_yaml = full_output / "kitti_100pct.yaml"
            full_report = prepare_dataset(
                root / "raw", full_output, fraction=1.0, train_ratio=0.80,
                seed=42, dataset_yaml_path=full_yaml,
            )
            self.assertEqual(full_report["train"]["images"], 16)
            self.assertEqual(full_report["val"]["images"], 4)
            self.assertEqual(full_report["dataset_yaml"], str(full_yaml.resolve()))
            self.assertTrue(full_yaml.is_file())


if __name__ == "__main__":
    unittest.main()

