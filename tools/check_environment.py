from __future__ import annotations

import json
import platform
import sys


def main() -> None:
    import cv2
    import torch
    import ultralytics

    cuda = torch.cuda.is_available()
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "opencv": cv2.__version__,
        "ultralytics": ultralytics.__version__,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not cuda:
        raise SystemExit("CUDA is unavailable. Install the GPU build of PyTorch first.")


if __name__ == "__main__":
    main()

