from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path

import pytesseract

from analyzer.agent_detector import AgentDetector
from analyzer.fixed_roi_kill_event_detector import FixedROIKillEventDetector
from analyzer.frame_sampler import sample_frames
from analyzer.roi_detector import ROIDetector
from analyzer.video_loader import VideoLoader
from config import (
    AGENT_DETECTION_MAX_FRAMES,
    OUTPUT_DIR,
    PLAYER_USERNAME,
    TESSERACT_CMD,
)


LOGGER = logging.getLogger("valorant-video-analyzer")
VIDEO_SUFFIXES = {".mp4"}
DEFAULT_SCAN_INTERVAL_SECONDS = 0.1
DEFAULT_KILL_BANNER_THRESHOLD = 0.12
DEFAULT_CLUSTER_GAP_SECONDS = 1.0
ANALYSIS_METHOD = "fixed_roi_kill_banner_feed_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze VALORANT videos with fixed ROIs. "
            "Folder input scans only that folder by default and generates fresh JSON output."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a gameplay file (.mp4) or a folder containing videos",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_SCAN_INTERVAL_SECONDS,
        help=f"Frame sampling interval in seconds (default: {DEFAULT_SCAN_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. If omitted, output is created near the input.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Filename to use for folder mode output (default: <folder_name>.json). "
            "Ignored when --output is provided."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subfolders recursively. By default, only the input folder is scanned.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N videos after sorting by filename (0 = all).",
    )
    parser.add_argument(
        "--kill-banner-threshold",
        type=float,
        default=DEFAULT_KILL_BANNER_THRESHOLD,
        help=(
            "Template match threshold in KILL_BANNER_BOX "
            f"(default: {DEFAULT_KILL_BANNER_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--cluster-gap",
        type=float,
        default=DEFAULT_CLUSTER_GAP_SECONDS,
        help=f"Seconds to merge repeated detections of the same kill (default: {DEFAULT_CLUSTER_GAP_SECONDS})",
    )
    parser.add_argument(
        "--debug-ocr",
        action="store_true",
        help=(
            "Enable kill-debug mode: show annotated frame window and save annotated "
            "images for each detected kill."
        ),
    )
    parser.add_argument(
        "--pause-on-kill",
        action="store_true",
        help=(
            "When debug mode is enabled, pause after each detected kill and wait "
            "for terminal input before continuing."
        ),
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=OUTPUT_DIR / "debug_kill_events",
        help="Directory to save annotated kill detection frames in debug mode.",
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


def _pick_primary_agent(values: list[str]) -> str:
    if not values:
        return "Unknown"
    known = [value for value in values if value and value != "Unknown"]
    if known:
        return Counter(known).most_common(1)[0][0]
    return values[0]


def _default_single_output_path(video_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{video_path.stem}.json"


def _default_folder_output_path(folder_path: Path, output_name: str | None) -> Path:
    filename = output_name or f"{folder_path.name}.json"
    return folder_path / filename


def discover_videos(folder_path: Path, recursive: bool = False) -> list[Path]:
    iterator = folder_path.rglob("*") if recursive else folder_path.glob("*")
    videos = [
        item
        for item in iterator
        if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
    ]
    return sorted(videos, key=lambda path: path.name.lower())


def analyze_video_with_fixed_rois(
    video_path: Path,
    interval_seconds: float,
    player_username: str,
    kill_banner_threshold: float,
    cluster_gap_seconds: float,
    debug_ocr: bool = False,
    pause_on_kill: bool = False,
    debug_dir: Path | None = None,
) -> dict:
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
    agent_detector = AgentDetector()
    fixed_roi_kill_detector = FixedROIKillEventDetector(
        player_username=player_username,
        kill_banner_threshold=kill_banner_threshold,
        cluster_gap_seconds=cluster_gap_seconds,
        debug_enabled=debug_ocr,
        debug_pause_on_kill=pause_on_kill,
        debug_show_window=debug_ocr,
        debug_output_dir=debug_dir,
    )

    step_for_agent = max(1, len(sampled_frames) // max(1, AGENT_DETECTION_MAX_FRAMES))
    agent_sampled = sampled_frames[::step_for_agent][:AGENT_DETECTION_MAX_FRAMES]
    agent_frames = [
        roi_detector.crop(frame, "agent_icon")
        for _, frame in agent_sampled
    ]
    hud_agent_name = agent_detector.detect_agent(agent_frames)

    events, feed_agent_name = fixed_roi_kill_detector.detect(
        sampled_frames,
        video_label=video_path.stem,
    )
    agent_name = hud_agent_name
    if agent_name == "Unknown" and feed_agent_name != "Unknown":
        LOGGER.info("Using kill-feed matched agent for %s: %s", video_path.name, feed_agent_name)
        agent_name = feed_agent_name

    event_details = [event.to_dict() for event in events]
    return {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "agent_name": agent_name,
        "kills": len(events),
        "kill_events": [{"timestamp": row["timestamp"]} for row in event_details],
        "kill_event_details": event_details,
    }


def process_folder_mode(
    folder_path: Path,
    interval_seconds: float,
    player_username: str,
    kill_banner_threshold: float,
    cluster_gap_seconds: float,
    recursive: bool,
    limit: int,
    debug_ocr: bool,
    pause_on_kill: bool,
    debug_dir: Path,
) -> dict:
    videos = discover_videos(folder_path=folder_path, recursive=recursive)
    if limit > 0:
        videos = videos[:limit]
    if not videos:
        raise RuntimeError(f"No .mp4 files found in folder: {folder_path}")

    LOGGER.info("Processing folder: %s (%d video(s))", folder_path, len(videos))
    video_results: list[dict] = []
    agent_votes: list[str] = []
    for video_path in videos:
        LOGGER.info("Analyzing video: %s", video_path.name)
        video_result = analyze_video_with_fixed_rois(
            video_path=video_path,
            interval_seconds=interval_seconds,
            player_username=player_username,
            kill_banner_threshold=kill_banner_threshold,
            cluster_gap_seconds=cluster_gap_seconds,
            debug_ocr=debug_ocr,
            pause_on_kill=pause_on_kill,
            debug_dir=debug_dir,
        )
        video_results.append(video_result)
        agent_votes.append(video_result["agent_name"])

    return {
        "folder": str(folder_path),
        "scan_recursive": recursive,
        "videos_processed": len(video_results),
        "agent_name": _pick_primary_agent(agent_votes),
        "analysis_method": ANALYSIS_METHOD,
        "videos": video_results,
    }


def write_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    if args.pause_on_kill and not args.debug_ocr:
        LOGGER.warning("--pause-on-kill requested without --debug-ocr. Enabling debug mode.")
        args.debug_ocr = True
    configure_tesseract()

    input_path = args.input_path.resolve()
    if not input_path.exists():
        LOGGER.error("Input path does not exist: %s", input_path)
        return 1

    if input_path.is_dir():
        try:
            result = process_folder_mode(
                folder_path=input_path,
                interval_seconds=args.interval,
                player_username=args.player,
                kill_banner_threshold=args.kill_banner_threshold,
                cluster_gap_seconds=args.cluster_gap,
                recursive=args.recursive,
                limit=args.limit,
                debug_ocr=args.debug_ocr,
                pause_on_kill=args.pause_on_kill,
                debug_dir=args.debug_dir.resolve(),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Folder analysis failed: %s", exc)
            return 1

        output_path = args.output.resolve() if args.output else _default_folder_output_path(
            folder_path=input_path,
            output_name=args.output_name,
        )
        write_json(result, output_path)
        result_with_output = {**result, "output_json": str(output_path)}
        print(json.dumps(result_with_output, indent=2))
        LOGGER.info("Folder JSON written to %s", output_path)
        return 0

    if not VideoLoader.is_valid_video(input_path):
        LOGGER.error("Video file does not exist or is not readable: %s", input_path)
        return 1

    try:
        result = analyze_video_with_fixed_rois(
            video_path=input_path,
            interval_seconds=args.interval,
            player_username=args.player,
            kill_banner_threshold=args.kill_banner_threshold,
            cluster_gap_seconds=args.cluster_gap,
            debug_ocr=args.debug_ocr,
            pause_on_kill=args.pause_on_kill,
            debug_dir=args.debug_dir.resolve(),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Analysis failed: %s", exc)
        return 1

    output_path = args.output.resolve() if args.output else _default_single_output_path(input_path)
    single_result = {
        **result,
        "analysis_method": ANALYSIS_METHOD,
    }
    write_json(single_result, output_path)
    print(json.dumps({**single_result, "output_json": str(output_path)}, indent=2))
    LOGGER.info("Analysis JSON written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
