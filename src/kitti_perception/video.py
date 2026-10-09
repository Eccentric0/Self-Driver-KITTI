from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def run_video(
    weights: Path,
    source: Path,
    output: Path,
    device: int | str = 0,
    confidence: float = 0.30,
    iou: float = 0.50,
    tracker: str = "bytetrack.yaml",
    trail_length: int = 30,
    codec: str = "mp4v",
) -> dict[str, Any]:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    weights = weights.expanduser().resolve()
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Detector weights not found: {weights}")
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    if len(codec) != 4:
        raise ValueError("Video codec must contain exactly four characters")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid input video dimensions: {width}x{height}")
    if fps <= 0:
        fps = 30.0
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV failed to create output video: {output}")

    model = YOLO(str(weights))
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=trail_length)
    )
    frames = 0
    inference_seconds = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            result = model.track(
                frame,
                persist=True,
                tracker=tracker,
                conf=confidence,
                iou=iou,
                device=device,
                verbose=False,
            )[0]
            inference_seconds += time.perf_counter() - started
            annotated = result.plot()
            boxes = result.boxes
            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                xywh = boxes.xywh.cpu().numpy()
                active = set(ids)
                for track_id, box in zip(ids, xywh):
                    center = (int(box[0]), int(box[1]))
                    trails[track_id].append(center)
                    points = np.asarray(trails[track_id], dtype=np.int32).reshape(-1, 1, 2)
                    if len(points) >= 2:
                        cv2.polylines(
                            annotated, [points], False, (0, 255, 255), 2, cv2.LINE_AA
                        )
                for stale_id in set(trails) - active:
                    trails.pop(stale_id, None)
            writer.write(annotated)
            frames += 1
    finally:
        capture.release()
        writer.release()

    if frames == 0:
        raise RuntimeError("No video frames were decoded")
    return {
        "source": str(source),
        "output": str(output),
        "frames": frames,
        "input_fps": fps,
        "average_inference_ms": 1000.0 * inference_seconds / frames,
        "average_inference_fps": frames / max(inference_seconds, 1e-9),
    }
