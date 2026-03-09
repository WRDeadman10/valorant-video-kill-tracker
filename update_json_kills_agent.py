from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import pytesseract

from analyzer.agent_detector import AgentDetector
from analyzer.fixed_roi_kill_event_detector import FixedROIKillEventDetector
from analyzer.frame_sampler import sample_frames
from analyzer.roi_detector import ROIDetector
from config import (
    AGENT_DETECTION_MAX_FRAMES,
    PLAYER_USERNAME,
    TESSERACT_CMD,
)


LOGGER = logging.getLogger("metadata-json-updater")
DEFAULT_METADATA_DIR = Path(r"E:\New folder\Valorant Tracker\shorts-uploader-engine\generated_metadata")
DEFAULT_VIDEOS_ROOT = Path(r"E:\New folder\Valorant Tracker\VALORANT")
DEFAULT_SCAN_INTERVAL_SECONDS = 0.1
DEFAULT_KILL_BANNER_THRESHOLD = 0.12
DEFAULT_CLUSTER_GAP_SECONDS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan metadata JSON files, resolve each relative_path video, detect kills + agent, "
            "and update the JSON in place."
        )
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help=f"Directory containing *.metadata.json (default: {DEFAULT_METADATA_DIR})",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=DEFAULT_VIDEOS_ROOT,
        help=f"Root folder that relative_path is appended to (default: {DEFAULT_VIDEOS_ROOT})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_SCAN_INTERVAL_SECONDS,
        help=(
            "Frame sampling interval in seconds for kill detection with fixed ROIs "
            f"(default: {DEFAULT_SCAN_INTERVAL_SECONDS})"
        ),
    )
    parser.add_argument(
        "--player",
        default=PLAYER_USERNAME,
        help=f'Player username for kill feed OCR matching (default: "{PLAYER_USERNAME}")',
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only process files where kills or agent_name is missing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max files to process (0 = all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and log changes without writing files.",
    )
    parser.add_argument(
        "--kill-banner-threshold",
        type=float,
        default=DEFAULT_KILL_BANNER_THRESHOLD,
        help=(
            "Template match threshold for KILL_BANNER_BOX "
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
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
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


def resolve_video_path(relative_path: str, videos_root: Path) -> Path | None:
    rel = str(relative_path or "").strip()
    if not rel:
        return None

    candidate_rel = Path(rel.replace("\\", "/"))
    if candidate_rel.is_absolute():
        return candidate_rel if candidate_rel.exists() else None

    candidate = (videos_root / candidate_rel).resolve()
    if candidate.exists():
        return candidate

    # Fallback search by basename if relative folder changed.
    fallback_name = candidate_rel.name
    if not fallback_name:
        return None

    matches = list(videos_root.rglob(fallback_name))
    if len(matches) == 1:
        return matches[0].resolve()
    return None


def detect_agent_and_kills(
    video_path: Path,
    interval_seconds: float,
    player_username: str,
    kill_banner_threshold: float,
    cluster_gap_seconds: float,
) -> tuple[str, list[dict]]:
    sampled_frames = sample_frames(video_path, interval_seconds=interval_seconds)
    if not sampled_frames:
        raise RuntimeError(f"No frames sampled from video: {video_path}")

    roi_detector = ROIDetector()
    agent_detector = AgentDetector()

    step_for_agent = max(1, len(sampled_frames) // max(1, AGENT_DETECTION_MAX_FRAMES))
    agent_sampled_frames = sampled_frames[::step_for_agent][:AGENT_DETECTION_MAX_FRAMES]
    agent_frames = [
        roi_detector.crop(frame, "agent_icon")
        for _, frame in agent_sampled_frames
    ]
    agent_name_hud = agent_detector.detect_agent(agent_frames)

    fixed_roi_detector = FixedROIKillEventDetector(
        player_username=player_username,
        kill_banner_threshold=kill_banner_threshold,
        cluster_gap_seconds=cluster_gap_seconds,
    )
    kill_events, feed_agent_name = fixed_roi_detector.detect(sampled_frames)

    agent_name = agent_name_hud
    if agent_name == "Unknown" and feed_agent_name != "Unknown":
        agent_name = feed_agent_name

    return agent_name, [event.to_dict() for event in kill_events]


def should_skip(entry: dict, only_missing: bool) -> bool:
    if not only_missing:
        return False

    kills = entry.get("kills")
    agent_name = str(entry.get("agent_name", "")).strip()
    if isinstance(kills, int) and kills >= 0 and agent_name:
        return True
    return False


def process_metadata_file(
    json_path: Path,
    videos_root: Path,
    interval_seconds: float,
    player_username: str,
    kill_banner_threshold: float,
    cluster_gap_seconds: float,
    only_missing: bool,
    dry_run: bool,
) -> tuple[str, str]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return "error", f"Invalid JSON ({json_path.name}): {exc}"

    if not isinstance(payload, dict):
        return "error", f"JSON root is not an object ({json_path.name})"

    if should_skip(payload, only_missing):
        return "skipped", f"Already has kills + agent_name ({json_path.name})"

    relative_path = str(payload.get("relative_path", "")).strip()
    if not relative_path:
        return "skipped", f"Missing relative_path ({json_path.name})"

    video_path = resolve_video_path(relative_path=relative_path, videos_root=videos_root)
    if video_path is None:
        return "skipped", f"Video not found for relative_path ({json_path.name})"

    try:
        agent_name, kill_events = detect_agent_and_kills(
            video_path=video_path,
            interval_seconds=interval_seconds,
            player_username=player_username,
            kill_banner_threshold=kill_banner_threshold,
            cluster_gap_seconds=cluster_gap_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        return "error", f"Analysis failed ({json_path.name}): {exc}"

    payload["kills"] = len(kill_events)
    payload["agent_name"] = agent_name
    payload["kill_events"] = [event["timestamp"] for event in kill_events]
    payload["kill_event_details"] = kill_events
    payload["analysis_method"] = "fixed_roi_kill_banner_feed_v2"
    payload["analysis_generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    if dry_run:
        return (
            "updated",
            (
                f"[dry-run] {json_path.name} -> kills={payload['kills']} "
                f"agent_name={payload['agent_name']}"
            ),
        )

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return (
        "updated",
        f"{json_path.name} -> kills={payload['kills']} agent_name={payload['agent_name']}",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    configure_tesseract()

    metadata_dir = args.metadata_dir.resolve()
    videos_root = args.videos_root.resolve()

    if not metadata_dir.exists() or not metadata_dir.is_dir():
        LOGGER.error("metadata-dir does not exist or is not a directory: %s", metadata_dir)
        return 1
    if not videos_root.exists() or not videos_root.is_dir():
        LOGGER.error("videos-root does not exist or is not a directory: %s", videos_root)
        return 1

    json_files = sorted(metadata_dir.glob("*.json"), key=lambda p: p.name.lower())
    if args.limit > 0:
        json_files = json_files[: args.limit]

    if not json_files:
        LOGGER.warning("No JSON files found in: %s", metadata_dir)
        return 0

    counts = {"updated": 0, "skipped": 0, "error": 0}
    for index, json_path in enumerate(json_files, start=1):
        LOGGER.info("[%d/%d] Processing %s", index, len(json_files), json_path.name)
        status, message = process_metadata_file(
            json_path=json_path,
            videos_root=videos_root,
            interval_seconds=args.interval,
            player_username=args.player,
            kill_banner_threshold=args.kill_banner_threshold,
            cluster_gap_seconds=args.cluster_gap,
            only_missing=args.only_missing,
            dry_run=args.dry_run,
        )
        counts[status] += 1
        if status == "updated":
            LOGGER.info(message)
        elif status == "skipped":
            LOGGER.warning(message)
        else:
            LOGGER.error(message)

    summary = {
        "metadata_dir": str(metadata_dir),
        "videos_root": str(videos_root),
        "dry_run": bool(args.dry_run),
        "processed": len(json_files),
        **counts,
    }
    print(json.dumps(summary, indent=2))
    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
