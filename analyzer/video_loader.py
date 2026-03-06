from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


class VideoLoader:
    @staticmethod
    def is_valid_video(video_path: str | Path) -> bool:
        path = Path(video_path)
        return path.exists() and path.is_file()

    @staticmethod
    def get_video_metadata(video_path: str | Path) -> VideoMetadata:
        path = Path(video_path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        duration_seconds = (frame_count / fps) if fps > 0 else 0.0
        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
        )

