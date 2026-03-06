from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
AGENT_ICON_DIR = ASSETS_DIR / "agent_icons"
TEMPLATE_DIR = ASSETS_DIR / "templates"
OUTPUT_DIR = PROJECT_ROOT / "output"


SAMPLE_INTERVAL_SECONDS = 0.5
MAP_OCR_MAX_FRAMES = 40
AGENT_DETECTION_MAX_FRAMES = 80


KNOWN_MAPS = [
    "Ascent",
    "Bind",
    "Haven",
    "Split",
    "Icebox",
    "Lotus",
    "Pearl",
    "Fracture",
    "Sunset",
]


AGENT_MATCH_THRESHOLD = 0.55
KILL_UI_TEMPLATE_FILENAME = "agent_name_v2.png"
KILL_UI_MATCH_THRESHOLD = 0.48
PLAYER_USERNAME = os.getenv("PLAYER_USERNAME", "Rangnar Lothbrok")
USERNAME_MATCH_THRESHOLD = 0.52
USERNAME_LEFT_MATCH_THRESHOLD = 0.56
USERNAME_LEFT_WEAK_THRESHOLD = 0.45
USERNAME_CENTER_MATCH_THRESHOLD = 0.70
KILL_CANDIDATE_CLUSTER_GAP_SECONDS = 3.0
KILL_CANDIDATE_MIN_HITS = 1
KILL_BACKGROUND_MAX_RED_RATIO = 0.18
KILLFEED_AGENT_ICON_MATCH_THRESHOLD = 0.24


TESSERACT_CMD = os.getenv("TESSERACT_CMD")


@dataclass(frozen=True)
class NormalizedROI:
    x: float
    y: float
    width: float
    height: float


ROI_CONFIG = {
    "map_name": NormalizedROI(x=0.02, y=0.02, width=0.22, height=0.08),
    "kill_feed": NormalizedROI(x=0.72, y=0.04, width=0.26, height=0.30),
    "agent_icon": NormalizedROI(x=0.43, y=0.82, width=0.14, height=0.15),
}
