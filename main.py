from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path

import pytesseract

from analyzer.agent_detector import AgentDetector
from analyzer.frame_sampler import sample_frames
from analyzer.json_exporter import build_video_result, export_folder_result, export_result
from analyzer.kill_feed_detector import KillFeedDetector
from analyzer.map_detector import MapDetector
from analyzer.roi_detector import ROIDetector
from analyzer.video_loader import VideoLoader
from config import (
    AGENT_DETECTION_MAX_FRAMES,
    MAP_OCR_MAX_FRAMES,
    OUTPUT_DIR,
    PLAYER_USERNAME,
    SAMPLE_INTERVAL_SECONDS,
    TESSERACT_CMD,
)


LOGGER = logging.getLogger("valorant-video-analyzer")
VIDEO_SUFFIXES = {".mp4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze VALORANT gameplay input. "
            "Input can be a single video file or a root folder scanned recursively."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to gameplay file (.mp4) or a root folder",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=SAMPLE_INTERVAL_SECONDS,
        help=f"Frame sampling interval in seconds (default: {SAMPLE_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "result.json",
        help="Output JSON path for single-file mode (default: output/result.json)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional fixed output JSON filename for each folder in directory mode. "
            "If omitted, each folder uses '<folder_name>.json'."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level",
    )
    parser.add_argument(
        "--player",
        default=PLAYER_USERNAME,
        help=f'Player username to detect in kill feed (default: "{PLAYER_USERNAME}")',
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def configure_tesseract() -> None:
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        LOGGER.info("Using Tesseract binary from TESSERACT_CMD: %s", TESSERACT_CMD)
        return

    default_windows_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default_windows_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(default_windows_path)
        LOGGER.info("Using Tesseract binary from default path: %s", default_windows_path)


def analyze_video(
    video_path: Path,
    interval_seconds: float,
    player_username: str,
) -> tuple[str, str, list[float]]:
    metadata = VideoLoader.get_video_metadata(video_path)
    LOGGER.info(
        "Video metadata (%s): %dx%d, %.2f fps, duration %.2fs",
        video_path.name,
        metadata.width,
        metadata.height,
        metadata.fps,
        metadata.duration_seconds,
    )

    sampled_frames = sample_frames(video_path, interval_seconds=interval_seconds)
    if not sampled_frames:
        raise RuntimeError(f"No frames were sampled from video: {video_path}")

    roi_detector = ROIDetector()
    map_detector = MapDetector()
    agent_detector = AgentDetector()
    kill_feed_detector = KillFeedDetector(player_username=player_username)

    map_frames = [
        roi_detector.crop(frame, "map_name")
        for _, frame in sampled_frames[:MAP_OCR_MAX_FRAMES]
    ]
    map_name = map_detector.detect_map(map_frames)

    agent_frames = [
        roi_detector.crop(frame, "agent_icon")
        for _, frame in sampled_frames[:AGENT_DETECTION_MAX_FRAMES]
    ]
    agent_name = agent_detector.detect_agent(agent_frames)

    kill_detection_interval = min(interval_seconds, 0.1)
    if kill_detection_interval < interval_seconds:
        LOGGER.info(
            "Running finer kill-feed pass for %s at %.1fs interval.",
            video_path.name,
            kill_detection_interval,
        )
        kill_frames = sample_frames(video_path, interval_seconds=kill_detection_interval)
    else:
        kill_frames = sampled_frames

    kill_timestamps = kill_feed_detector.detect_kills(kill_frames, roi_detector)
    killfeed_agent_name = kill_feed_detector.last_detected_agent_name
    if agent_name == "Unknown" and killfeed_agent_name != "Unknown":
        LOGGER.info("Using kill-feed matched agent for %s: %s", video_path.name, killfeed_agent_name)
        agent_name = killfeed_agent_name

    return map_name, agent_name, kill_timestamps


def discover_videos_by_folder(root_path: Path) -> dict[Path, list[Path]]:
    videos_by_folder: dict[Path, list[Path]] = defaultdict(list)
    for candidate in root_path.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        videos_by_folder[candidate.parent].append(candidate)

    for folder, videos in list(videos_by_folder.items()):
        videos_by_folder[folder] = sorted(videos, key=lambda path: path.name.lower())
    return dict(sorted(videos_by_folder.items(), key=lambda item: str(item[0]).lower()))


def pick_shared_label(values: list[str]) -> str:
    if not values:
        return "Unknown"
    known_values = [value for value in values if value and value != "Unknown"]
    if known_values:
        return Counter(known_values).most_common(1)[0][0]
    return values[0]


def process_directory_mode(
    root_path: Path,
    interval_seconds: float,
    player_username: str,
    output_name: str | None,
) -> list[dict]:
    videos_by_folder = discover_videos_by_folder(root_path)
    if not videos_by_folder:
        raise RuntimeError(f"No .mp4 files found under: {root_path}")

    summaries: list[dict] = []
    for folder, videos in videos_by_folder.items():
        LOGGER.info("Processing folder: %s (%d video(s))", folder, len(videos))
        folder_video_results: list[dict] = []
        map_votes: list[str] = []
        agent_votes: list[str] = []

        for video_path in videos:
            LOGGER.info("Analyzing video: %s", video_path.name)
            map_name, agent_name, kill_timestamps = analyze_video(
                video_path=video_path,
                interval_seconds=interval_seconds,
                player_username=player_username,
            )
            map_votes.append(map_name)
            agent_votes.append(agent_name)
            folder_video_results.append(
                build_video_result(
                    video_path=video_path,
                    map_name=map_name,
                    agent_name=agent_name,
                    kill_events=kill_timestamps,
                )
            )

        shared_map = pick_shared_label(map_votes)
        shared_agent = pick_shared_label(agent_votes)
        folder_output_name = output_name or f"{folder.name}.json"
        output_path = folder / folder_output_name
        export_folder_result(
            map_name=shared_map,
            agent_name=shared_agent,
            video_results=folder_video_results,
            output_path=output_path,
        )
        LOGGER.info("Folder JSON written: %s", output_path)

        summaries.append(
            {
                "folder": str(folder),
                "output_json": str(output_path),
                "videos_processed": len(folder_video_results),
                "map": shared_map,
                "agent": shared_agent,
            }
        )

    return summaries


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    configure_tesseract()

    input_path = args.input_path.resolve()
    if not input_path.exists():
        LOGGER.error("Input path does not exist: %s", input_path)
        return 1

    if input_path.is_dir():
        try:
            summaries = process_directory_mode(
                root_path=input_path,
                interval_seconds=args.interval,
                player_username=args.player,
                output_name=args.output_name,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Directory analysis failed: %s", exc)
            return 1

        print(
            json.dumps(
                {
                    "root": str(input_path),
                    "folders_processed": len(summaries),
                    "folders": summaries,
                },
                indent=2,
            )
        )
        return 0

    if not VideoLoader.is_valid_video(input_path):
        LOGGER.error("Video file does not exist or is not readable: %s", input_path)
        return 1

    try:
        map_name, agent_name, kill_timestamps = analyze_video(
            video_path=input_path,
            interval_seconds=args.interval,
            player_username=args.player,
        )
        result = export_result(
            video_path=input_path,
            map_name=map_name,
            agent_name=agent_name,
            kill_events=kill_timestamps,
            output_path=args.output,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Analysis failed: %s", exc)
        return 1

    print(json.dumps(result, indent=2))
    LOGGER.info("Analysis JSON written to %s", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
