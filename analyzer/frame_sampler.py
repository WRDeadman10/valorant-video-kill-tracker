from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import ffmpeg
import numpy as np
from tqdm import tqdm

from config import SAMPLE_INTERVAL_SECONDS


def extract_frames_ffmpeg(
    video_path: str | Path,
    output_dir: str | Path,
    interval_seconds: float,
) -> list[Path]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = 1.0 / interval_seconds
    output_pattern = str(output_dir / "frame_%06d.jpg")
    stream = ffmpeg.input(str(video_path))
    stream = ffmpeg.output(
        stream,
        output_pattern,
        vf=f"fps={fps}",
        qscale=2,
        loglevel="error",
    )

    try:
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"FFmpeg extraction failed: {stderr}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg binary not found. Install FFmpeg and add it to PATH.") from exc

    return sorted(output_dir.glob("frame_*.jpg"))


def sample_frames(
    video_path: str | Path,
    interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
) -> list[tuple[float, np.ndarray]]:
    sampled: list[tuple[float, np.ndarray]] = []

    with tempfile.TemporaryDirectory(prefix="valorant_frames_") as tmp_dir:
        frame_paths = extract_frames_ffmpeg(video_path, tmp_dir, interval_seconds)
        for idx, frame_path in enumerate(
            tqdm(frame_paths, desc="Loading sampled frames", unit="frame")
        ):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            timestamp = round(idx * interval_seconds, 3)
            sampled.append((timestamp, frame))

    return sampled
