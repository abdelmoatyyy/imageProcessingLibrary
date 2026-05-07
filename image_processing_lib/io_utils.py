"""Save outputs under project `output-images/<task_id>/`."""

from pathlib import Path

import cv2


def save_output(image, task_id: str, filename: str) -> str:
    out_dir = Path(__file__).resolve().parent.parent / "output-images" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    cv2.imwrite(str(path), image)
    return str(path)
