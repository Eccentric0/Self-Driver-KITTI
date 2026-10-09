import tempfile
import unittest
from pathlib import Path

from kitti_perception.config import load_config, project_path


class ConfigTests(unittest.TestCase):
    def test_project_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            path = config_dir / "kitti.yaml"
            path.write_text("data: {}\n", encoding="utf-8")
            config = load_config(path)
            self.assertEqual(project_path(config, "data/kitti"), root / "data" / "kitti")


if __name__ == "__main__":
    unittest.main()

