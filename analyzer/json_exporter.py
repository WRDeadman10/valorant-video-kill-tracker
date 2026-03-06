from __future__ import annotations

import json
from pathlib import Path


def _write_json(data: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _format_kill_events(kill_events: list[float]) -> list[dict]:
    return [{"timestamp": round(timestamp, 1)} for timestamp in kill_events]


def build_video_result(
    video_path: str | Path,
    map_name: str,
    agent_name: str,
    kill_events: list[float],
) -> dict:
    return {
        "video": Path(video_path).name,
        "map": map_name,
        "agent": agent_name,
        "kills": len(kill_events),
        "kill_events": _format_kill_events(kill_events),
    }


def export_result(
    video_path: str | Path,
    map_name: str,
    agent_name: str,
    kill_events: list[float],
    output_path: str | Path,
) -> dict:
    result = build_video_result(
        video_path=video_path,
        map_name=map_name,
        agent_name=agent_name,
        kill_events=kill_events,
    )
    _write_json(result, output_path)
    return result


def build_folder_result(
    map_name: str,
    agent_name: str,
    video_results: list[dict],
) -> dict:
    videos = [
        {
            "video_name": item["video"],
            "kills": item["kills"],
            "kill_events": item["kill_events"],
        }
        for item in video_results
    ]
    return {
        "map": map_name,
        "agent": agent_name,
        "videos": videos,
    }


def export_folder_result(
    map_name: str,
    agent_name: str,
    video_results: list[dict],
    output_path: str | Path,
) -> dict:
    result = build_folder_result(
        map_name=map_name,
        agent_name=agent_name,
        video_results=video_results,
    )
    _write_json(result, output_path)
    return result
