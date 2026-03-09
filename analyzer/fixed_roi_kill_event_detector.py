from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import difflib
import logging
from pathlib import Path
import re

import cv2
import numpy as np
import pytesseract

from analyzer.kill_feed_detector import KillFeedDetector
from config import TEMPLATE_DIR


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FixedROI:
    x: int
    y: int
    w: int
    h: int


DEFAULT_FIXED_ROIS: dict[str, FixedROI] = {
    "SITE_BOX": FixedROI(x=0, y=0, w=350, h=120),
    "KILL_FEED_BOX": FixedROI(x=1500, y=0, w=420, h=250),
    "KILL_BANNER_BOX": FixedROI(x=760, y=850, w=400, h=150),
    "WEAPON_BOX": FixedROI(x=1600, y=880, w=250, h=200),
}

BASE_WIDTH = 1920
BASE_HEIGHT = 1080

KNOWN_WEAPONS = [
    "Classic",
    "Shorty",
    "Frenzy",
    "Ghost",
    "Sheriff",
    "Stinger",
    "Spectre",
    "Bucky",
    "Judge",
    "Bulldog",
    "Guardian",
    "Phantom",
    "Vandal",
    "Marshal",
    "Outlaw",
    "Operator",
    "Ares",
    "Odin",
    "Knife",
]

KNOWN_SITE_NAMES = [
    "A SITE",
    "B SITE",
    "C SITE",
    "A MAIN",
    "B MAIN",
    "C MAIN",
    "MID",
    "ATTACKER SPAWN",
    "DEFENDER SPAWN",
    "A",
    "B",
    "C",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return safe or "video"


class FixedROICropper:
    def __init__(
        self,
        rois: dict[str, FixedROI] | None = None,
        base_width: int = BASE_WIDTH,
        base_height: int = BASE_HEIGHT,
    ) -> None:
        self.rois = rois or DEFAULT_FIXED_ROIS
        self.base_width = base_width
        self.base_height = base_height

    def crop(self, frame: np.ndarray, roi_name: str) -> np.ndarray:
        if roi_name not in self.rois:
            raise KeyError(f"Unknown ROI: {roi_name}")

        roi = self.rois[roi_name]
        frame_h, frame_w = frame.shape[:2]
        scale_x = frame_w / float(self.base_width)
        scale_y = frame_h / float(self.base_height)

        x = max(0, int(round(roi.x * scale_x)))
        y = max(0, int(round(roi.y * scale_y)))
        w = max(1, int(round(roi.w * scale_x)))
        h = max(1, int(round(roi.h * scale_y)))

        x1 = min(frame_w, x + w)
        y1 = min(frame_h, y + h)
        return frame[y:y1, x:x1]

    def pixel_roi(self, frame: np.ndarray, roi_name: str) -> tuple[int, int, int, int]:
        if roi_name not in self.rois:
            raise KeyError(f"Unknown ROI: {roi_name}")

        roi = self.rois[roi_name]
        frame_h, frame_w = frame.shape[:2]
        scale_x = frame_w / float(self.base_width)
        scale_y = frame_h / float(self.base_height)

        x = max(0, int(round(roi.x * scale_x)))
        y = max(0, int(round(roi.y * scale_y)))
        w = max(1, int(round(roi.w * scale_x)))
        h = max(1, int(round(roi.h * scale_y)))

        x1 = min(frame_w, x + w)
        y1 = min(frame_h, y + h)
        return x, y, max(1, x1 - x), max(1, y1 - y)


class KillBannerDetector:
    def __init__(
        self,
        template_dirs: list[Path] | None = None,
        edge_weight: float = 0.9,
    ) -> None:
        if template_dirs is None:
            template_dirs = [Path(TEMPLATE_DIR), Path(TEMPLATE_DIR).parent / "templets"]
        self.template_dirs = template_dirs
        self.edge_weight = edge_weight
        self.templates = self._load_templates()

    @staticmethod
    def _prepare_template(image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if image is None or image.size == 0:
            return None

        if image.ndim == 3 and image.shape[2] == 4:
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_and(gray, gray, mask=alpha)
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.equalizeHist(gray)
        edge = cv2.Canny(gray, 60, 150)
        return gray, edge

    def _load_templates(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        templates: list[tuple[str, np.ndarray, np.ndarray]] = []
        seen_paths: set[Path] = set()
        for directory in self.template_dirs:
            if not directory.exists():
                continue
            for pattern in ("*Kill_Banner*.png", "*Kill_Banner*.webp", "*Kill_Banner*.jpg", "*Kill_Banner*.jpeg", "*Kill_Banner*.bmp"):
                for template_path in sorted(directory.glob(pattern)):
                    resolved = template_path.resolve()
                    if resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)
                    image = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
                    prepared = self._prepare_template(image)
                    if prepared is None:
                        continue
                    templates.append((template_path.name, prepared[0], prepared[1]))

        if templates:
            LOGGER.info("Loaded %d kill banner templates.", len(templates))
        else:
            LOGGER.warning("No kill banner templates found in configured template directories.")
        return templates

    @staticmethod
    def _prepare_crop(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.equalizeHist(gray)
        edge = cv2.Canny(gray, 60, 150)
        return gray, edge

    def score(self, banner_crop: np.ndarray) -> tuple[str, float]:
        if banner_crop is None or banner_crop.size == 0:
            return "Unknown", 0.0
        if not self.templates:
            return "Unknown", 0.0

        crop_gray, crop_edge = self._prepare_crop(banner_crop)
        crop_h, crop_w = crop_gray.shape[:2]
        if crop_h < 6 or crop_w < 6:
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = 0.0
        for name, tpl_gray, tpl_edge in self.templates:
            tpl_h, tpl_w = tpl_gray.shape[:2]
            if tpl_h > crop_h or tpl_w > crop_w:
                fit_gray = cv2.resize(tpl_gray, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
                fit_edge = cv2.resize(tpl_edge, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
            else:
                fit_gray = tpl_gray
                fit_edge = tpl_edge

            gray_result = cv2.matchTemplate(crop_gray, fit_gray, cv2.TM_CCOEFF_NORMED)
            _, gray_score, _, _ = cv2.minMaxLoc(gray_result)

            edge_result = cv2.matchTemplate(crop_edge, fit_edge, cv2.TM_CCOEFF_NORMED)
            _, edge_score, _, _ = cv2.minMaxLoc(edge_result)

            combined = float(max(gray_score, edge_score * self.edge_weight))
            if combined > best_score:
                best_score = combined
                best_name = name

        return best_name, best_score


class OCRExtractor:
    def __init__(self) -> None:
        self._ocr_available = True
        self._warned_missing_tesseract = False

    @staticmethod
    def _prepare_variants(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(enlarged, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = cv2.bitwise_not(binary)
        return binary, inverted

    def extract_text(self, crop: np.ndarray, psm: int = 7) -> str:
        if not self._ocr_available:
            return ""
        if crop is None or crop.size == 0:
            return ""

        binary, inverted = self._prepare_variants(crop)
        config = f"--oem 3 --psm {psm}"
        try:
            text_a = pytesseract.image_to_string(binary, config=config)
            text_b = pytesseract.image_to_string(inverted, config=config)
        except pytesseract.pytesseract.TesseractNotFoundError:
            self._ocr_available = False
            if not self._warned_missing_tesseract:
                LOGGER.warning("Tesseract not found. Site/weapon OCR disabled.")
                self._warned_missing_tesseract = True
            return ""
        except pytesseract.pytesseract.TesseractError:
            return ""

        merged = " ".join((text_a, text_b)).replace("\n", " ").strip()
        merged = re.sub(r"\s+", " ", merged)
        return merged

    @staticmethod
    def _cleanup_text(text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9 ]", " ", text).upper()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def extract_site_info(self, site_crop: np.ndarray) -> tuple[str, str]:
        raw = self.extract_text(site_crop, psm=7)
        cleaned = self._cleanup_text(raw)
        if not cleaned:
            return "Unknown", raw

        cleaned_norm = _normalize_text(cleaned)
        best_site = "Unknown"
        best_score = 0.0
        for candidate in KNOWN_SITE_NAMES:
            cand_norm = _normalize_text(candidate)
            if not cand_norm:
                continue
            if cand_norm in cleaned_norm or cleaned_norm in cand_norm:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, cand_norm, cleaned_norm).ratio()
            if score > best_score:
                best_score = score
                best_site = candidate

        if best_score >= 0.62:
            return best_site, raw
        return "Unknown", raw

    def extract_weapon_info(self, weapon_crop: np.ndarray) -> tuple[str, str]:
        raw = self.extract_text(weapon_crop, psm=6)
        cleaned = self._cleanup_text(raw)
        if not cleaned:
            return "Unknown", raw

        cleaned_norm = _normalize_text(cleaned)
        best_weapon = "Unknown"
        best_score = 0.0
        for weapon in KNOWN_WEAPONS:
            weapon_norm = _normalize_text(weapon)
            if weapon_norm in cleaned_norm or cleaned_norm in weapon_norm:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, weapon_norm, cleaned_norm).ratio()
            if score > best_score:
                best_score = score
                best_weapon = weapon

        if best_score >= 0.60:
            return best_weapon, raw
        return "Unknown", raw

    def extract_site_name(self, site_crop: np.ndarray) -> str:
        return self.extract_site_info(site_crop)[0]

    def extract_weapon_name(self, weapon_crop: np.ndarray) -> str:
        return self.extract_weapon_info(weapon_crop)[0]


@dataclass(frozen=True)
class KillEvent:
    timestamp: float
    site_name: str
    weapon_used: str
    agent_name: str
    kill_banner_template: str
    kill_banner_score: float
    username_left_score: float
    kill_feed_ui_score: float

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "site_name": self.site_name,
            "weapon_used": self.weapon_used,
            "agent_name": self.agent_name,
            "kill_banner_template": self.kill_banner_template,
            "kill_banner_score": round(self.kill_banner_score, 3),
            "username_left_score": round(self.username_left_score, 3),
            "kill_feed_ui_score": round(self.kill_feed_ui_score, 3),
        }


class FixedROIKillEventDetector:
    def __init__(
        self,
        player_username: str,
        kill_banner_threshold: float = 0.12,
        cluster_gap_seconds: float = 1.0,
        template_dir: str | Path = TEMPLATE_DIR,
        debug_enabled: bool = False,
        debug_pause_on_kill: bool = False,
        debug_show_window: bool = True,
        debug_output_dir: str | Path | None = None,
    ) -> None:
        self.kill_banner_threshold = kill_banner_threshold
        self.cluster_gap_seconds = cluster_gap_seconds
        self.cropper = FixedROICropper()
        self.kill_feed_detector = KillFeedDetector(
            template_dir=template_dir,
            player_username=player_username,
        )
        self.banner_detector = KillBannerDetector(
            template_dirs=[Path(template_dir), Path(template_dir).parent / "templets"]
        )
        self.ocr = OCRExtractor()
        self.debug_enabled = debug_enabled
        self.debug_pause_on_kill = debug_pause_on_kill
        self.debug_show_window = debug_show_window
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir else Path("output") / "debug_kill_events"
        self._debug_window_name = "VALORANT Kill OCR Debug"
        self._debug_window_available = True
        self._stop_requested = False

    def _cluster_candidates(self, candidates: list[dict]) -> list[list[dict]]:
        if not candidates:
            return []
        clusters: list[list[dict]] = [[candidates[0]]]
        for candidate in candidates[1:]:
            previous = clusters[-1][-1]
            if candidate["timestamp"] - previous["timestamp"] <= self.cluster_gap_seconds:
                clusters[-1].append(candidate)
            else:
                clusters.append([candidate])
        return clusters

    def _annotate_frame(self, frame: np.ndarray, row: dict) -> np.ndarray:
        annotated = frame.copy()

        roi_colors = {
            "SITE_BOX": (0, 255, 255),
            "KILL_FEED_BOX": (0, 255, 0),
            "KILL_BANNER_BOX": (255, 200, 0),
            "WEAPON_BOX": (255, 0, 255),
        }
        for roi_name, color in roi_colors.items():
            x, y, w, h = self.cropper.pixel_roi(annotated, roi_name)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                annotated,
                roi_name,
                (x + 4, max(18, y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        lines = [
            f"time: {row['timestamp']:.3f}s",
            f"banner: {row['kill_banner_template']} ({row['kill_banner_score']:.3f})",
            f"kill_feed_ui: {row['kill_feed_ui_score']:.3f}  left_user_score: {row['username_left_score']:.3f}",
            f"agent: {row['agent_name']}  site: {row['site_name']}  weapon: {row['weapon_used']}",
            f"site_ocr_raw: {str(row.get('site_ocr_raw', ''))[:80]}",
            f"weapon_ocr_raw: {str(row.get('weapon_ocr_raw', ''))[:80]}",
            f"kill_feed_text: {str(row.get('matched_text', ''))[:80]}",
        ]

        y = 24
        for line in lines:
            text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            tw, th = text_size
            cv2.rectangle(
                annotated,
                (8, y - th - 8),
                (14 + tw, y + 6),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                annotated,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 24

        return annotated

    def _save_debug_frame(
        self,
        annotated_frame: np.ndarray,
        video_label: str,
        event_index: int,
        timestamp: float,
    ) -> Path | None:
        try:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{_sanitize_filename(video_label)}"
                f"_kill_{event_index:03d}_{timestamp:.3f}s.png"
            )
            output_path = self.debug_output_dir / filename
            cv2.imwrite(str(output_path), annotated_frame)
            return output_path
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to save debug frame: %s", exc)
            return None

    def _show_debug_window(self, annotated_frame: np.ndarray) -> None:
        if not self.debug_show_window or not self._debug_window_available:
            return
        try:
            display = annotated_frame
            if annotated_frame.shape[1] > 1600:
                scale = 1600.0 / annotated_frame.shape[1]
                display = cv2.resize(
                    annotated_frame,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            cv2.namedWindow(self._debug_window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(self._debug_window_name, display)
            cv2.waitKey(1)
        except cv2.error as exc:
            self._debug_window_available = False
            LOGGER.warning("OpenCV debug window not available: %s", exc)

    def _pause_for_user(self, event_index: int, timestamp: float, saved_path: Path | None) -> None:
        if not self.debug_pause_on_kill:
            return
        location = str(saved_path) if saved_path else "(not saved)"
        prompt = (
            f"\n[DEBUG] Kill #{event_index} detected at {timestamp:.3f}s.\n"
            f"[DEBUG] Frame: {location}\n"
            "[DEBUG] Press Enter for next kill, or type 'q' then Enter to stop debug run: "
        )
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return
        if answer == "q":
            self._stop_requested = True

    def detect(
        self,
        sampled_frames: list[tuple[float, np.ndarray]],
        video_label: str | None = None,
    ) -> tuple[list[KillEvent], str]:
        if not sampled_frames:
            return [], "Unknown"
        if self.kill_feed_detector.ui_template is None:
            LOGGER.warning("Kill feed template unavailable. Kill detection skipped.")
            return [], "Unknown"

        self._stop_requested = False
        label = video_label or "video"
        candidates: list[dict] = []
        agent_votes: list[str] = []
        for timestamp, frame in sampled_frames:
            if self._stop_requested:
                break
            kill_feed_crop = self.cropper.crop(frame, "KILL_FEED_BOX")
            if kill_feed_crop is None or kill_feed_crop.size == 0:
                continue

            kill_feed_gray = self.kill_feed_detector._preprocess(kill_feed_crop)
            ui_score, top_left, match_size = self.kill_feed_detector._match_template_in_feed(
                kill_feed_gray, self.kill_feed_detector.ui_template
            )
            if ui_score < self.kill_feed_detector.ui_match_threshold:
                continue

            row_crop = self.kill_feed_detector._crop_with_padding(
                kill_feed_crop,
                x=top_left[0],
                y=top_left[1],
                w=match_size[0],
                h=match_size[1],
            )
            (
                is_candidate,
                _confidence,
                scores,
                matched_text,
                _red_ratio,
                agent_name,
                _agent_score,
            ) = self.kill_feed_detector._classify_candidate(row_crop)
            if not is_candidate:
                continue

            left_score = float(scores.get("left", 0.0))
            if left_score < self.kill_feed_detector.username_left_weak_threshold:
                continue

            kill_banner_crop = self.cropper.crop(frame, "KILL_BANNER_BOX")
            banner_template, banner_score = self.banner_detector.score(kill_banner_crop)
            if banner_score < self.kill_banner_threshold:
                continue

            site_crop = self.cropper.crop(frame, "SITE_BOX")
            weapon_crop = self.cropper.crop(frame, "WEAPON_BOX")
            site_name, site_raw = self.ocr.extract_site_info(site_crop)
            weapon_used, weapon_raw = self.ocr.extract_weapon_info(weapon_crop)

            candidates.append(
                {
                    "timestamp": float(timestamp),
                    "site_name": site_name,
                    "site_ocr_raw": site_raw,
                    "weapon_used": weapon_used,
                    "weapon_ocr_raw": weapon_raw,
                    "agent_name": agent_name,
                    "kill_banner_template": banner_template,
                    "kill_banner_score": float(banner_score),
                    "username_left_score": left_score,
                    "kill_feed_ui_score": float(ui_score),
                    "matched_text": matched_text,
                    "frame": frame.copy(),
                }
            )
            if agent_name != "Unknown":
                agent_votes.append(agent_name)

        clusters = self._cluster_candidates(candidates)
        events: list[KillEvent] = []
        for event_index, cluster in enumerate(clusters, start=1):
            if self._stop_requested:
                break
            representative = max(
                cluster,
                key=lambda row: (
                    row["kill_banner_score"] * 1.4
                    + row["username_left_score"]
                    + row["kill_feed_ui_score"] * 0.5
                ),
            )
            if self.debug_enabled:
                annotated = self._annotate_frame(representative["frame"], representative)
                saved_path = self._save_debug_frame(
                    annotated_frame=annotated,
                    video_label=label,
                    event_index=event_index,
                    timestamp=representative["timestamp"],
                )
                self._show_debug_window(annotated)
                self._pause_for_user(
                    event_index=event_index,
                    timestamp=representative["timestamp"],
                    saved_path=saved_path,
                )
            events.append(
                KillEvent(
                    timestamp=representative["timestamp"],
                    site_name=representative["site_name"],
                    weapon_used=representative["weapon_used"],
                    agent_name=representative["agent_name"],
                    kill_banner_template=representative["kill_banner_template"],
                    kill_banner_score=representative["kill_banner_score"],
                    username_left_score=representative["username_left_score"],
                    kill_feed_ui_score=representative["kill_feed_ui_score"],
                )
            )

        feed_agent = "Unknown"
        if agent_votes:
            feed_agent = Counter(agent_votes).most_common(1)[0][0]
        if self.debug_show_window and self._debug_window_available:
            try:
                cv2.destroyWindow(self._debug_window_name)
            except cv2.error:
                pass
        return events, feed_agent
