from __future__ import annotations

from collections import Counter
import difflib
import logging
from pathlib import Path
import re

import cv2
import numpy as np
import pytesseract

from config import (
    AGENT_ICON_DIR,
    KILL_BACKGROUND_MAX_RED_RATIO,
    KILL_CANDIDATE_CLUSTER_GAP_SECONDS,
    KILL_CANDIDATE_MIN_HITS,
    KILL_UI_MATCH_THRESHOLD,
    KILL_UI_TEMPLATE_FILENAME,
    KILLFEED_AGENT_ICON_MATCH_THRESHOLD,
    PLAYER_USERNAME,
    TEMPLATE_DIR,
    USERNAME_CENTER_MATCH_THRESHOLD,
    USERNAME_LEFT_MATCH_THRESHOLD,
    USERNAME_LEFT_WEAK_THRESHOLD,
    USERNAME_MATCH_THRESHOLD,
)


LOGGER = logging.getLogger(__name__)


class KillFeedDetector:
    def __init__(
        self,
        template_dir: str | Path = TEMPLATE_DIR,
        player_username: str = PLAYER_USERNAME,
        ui_template_name: str = KILL_UI_TEMPLATE_FILENAME,
        ui_match_threshold: float = KILL_UI_MATCH_THRESHOLD,
        username_match_threshold: float = USERNAME_MATCH_THRESHOLD,
        username_left_match_threshold: float = USERNAME_LEFT_MATCH_THRESHOLD,
        username_left_weak_threshold: float = USERNAME_LEFT_WEAK_THRESHOLD,
        username_center_match_threshold: float = USERNAME_CENTER_MATCH_THRESHOLD,
        cluster_gap_seconds: float = KILL_CANDIDATE_CLUSTER_GAP_SECONDS,
        min_cluster_hits: int = KILL_CANDIDATE_MIN_HITS,
        non_red_max_ratio: float = KILL_BACKGROUND_MAX_RED_RATIO,
        killfeed_agent_threshold: float = KILLFEED_AGENT_ICON_MATCH_THRESHOLD,
    ) -> None:
        self.template_dir = Path(template_dir)
        self.player_username = player_username
        self.player_username_norm = self._normalize_text(player_username)
        self.player_username_tokens = [
            self._normalize_text(token)
            for token in re.split(r"\s+", player_username)
            if self._normalize_text(token)
        ]
        self.ui_match_threshold = ui_match_threshold
        self.username_match_threshold = username_match_threshold
        self.username_left_match_threshold = username_left_match_threshold
        self.username_left_weak_threshold = username_left_weak_threshold
        self.username_center_match_threshold = username_center_match_threshold
        self.cluster_gap_seconds = cluster_gap_seconds
        self.min_cluster_hits = max(1, min_cluster_hits)
        self.non_red_max_ratio = non_red_max_ratio
        self.killfeed_agent_threshold = killfeed_agent_threshold

        self._ocr_available = True
        self._warned_missing_tesseract = False
        self.last_detected_agent_name = "Unknown"

        self.ui_template = self._load_ui_template(ui_template_name)
        self.killfeed_agent_templates = self._load_killfeed_agent_templates()

    def _candidate_template_paths(self, filename: str) -> list[Path]:
        dirs = [
            self.template_dir,
            self.template_dir.parent / "templets",  # User-provided typo path compatibility
        ]
        return [directory / filename for directory in dirs]

    def _load_ui_template(self, ui_template_name: str) -> np.ndarray | None:
        candidates = self._candidate_template_paths(ui_template_name)
        # Backward-compatible fallback
        if ui_template_name != "agent_name.png":
            candidates.extend(self._candidate_template_paths("agent_name.png"))

        for path in candidates:
            if not path.exists():
                continue
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is not None and template.size > 0:
                LOGGER.info("Using kill-feed template: %s", path)
                return template

        LOGGER.warning("Kill UI template not found. Checked: %s", ", ".join(str(p) for p in candidates))
        return None

    def _candidate_agent_icon_dirs(self) -> list[Path]:
        dirs = [
            Path(AGENT_ICON_DIR),
            self.template_dir.parent / "agents" / "agetns_icon",  # User-provided typo path compatibility
            self.template_dir.parent / "agents" / "agent_icons",
            self.template_dir.parent / "agent_icons",
        ]
        existing: list[Path] = []
        seen: set[Path] = set()
        for directory in dirs:
            resolved = directory.resolve()
            if resolved in seen or not directory.exists():
                continue
            seen.add(resolved)
            existing.append(directory)
        return existing

    @staticmethod
    def _clean_agent_name(name: str) -> str:
        normalized = re.sub(r"[_\-\s]*icon$", "", name, flags=re.IGNORECASE)
        normalized = normalized.replace("_", " ").replace("-", " ").strip()
        return normalized.title() if normalized else "Unknown"

    @staticmethod
    def _prepare_icon_template(image: np.ndarray) -> np.ndarray | None:
        if image is None or image.size == 0:
            return None

        if image.ndim == 3 and image.shape[2] == 4:
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
            ys, xs = np.where(alpha > 10)
            if len(xs) and len(ys):
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                bgr = bgr[y0:y1, x0:x1]
                alpha = alpha[y0:y1, x0:x1]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_and(gray, gray, mask=alpha)
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if gray is None or gray.size == 0:
            return None

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    def _load_killfeed_agent_templates(self) -> dict[str, np.ndarray]:
        templates: dict[str, np.ndarray] = {}
        for directory in self._candidate_agent_icon_dirs():
            for pattern in ("*.webp", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
                for template_path in sorted(directory.glob(pattern)):
                    image = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
                    prepared = self._prepare_icon_template(image)
                    if prepared is None:
                        continue
                    agent_name = self._clean_agent_name(template_path.stem)
                    templates[agent_name] = prepared

        if not templates:
            LOGGER.warning("No kill-feed agent icon templates were loaded.")
        else:
            LOGGER.info("Loaded %d kill-feed agent icon templates.", len(templates))
        return templates

    @staticmethod
    def _preprocess(crop) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    @staticmethod
    def _best_substring_ratio(target: str, text: str) -> float:
        if not target or not text:
            return 0.0
        if len(text) < len(target):
            return difflib.SequenceMatcher(None, target, text).ratio()

        window = len(target)
        best = 0.0
        for start in range(0, len(text) - window + 1):
            ratio = difflib.SequenceMatcher(None, target, text[start : start + window]).ratio()
            if ratio > best:
                best = ratio
        return best

    @staticmethod
    def _match_template_in_feed(
        kill_feed_gray: np.ndarray,
        template: np.ndarray,
    ) -> tuple[float, tuple[int, int], tuple[int, int]]:
        feed_h, feed_w = kill_feed_gray.shape[:2]
        tpl_h, tpl_w = template.shape[:2]

        if tpl_h > feed_h or tpl_w > feed_w:
            if feed_h < 2 or feed_w < 2:
                return 0.0, (0, 0), (0, 0)
            resized_tpl = cv2.resize(template, (feed_w, feed_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(kill_feed_gray, resized_tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, top_left = cv2.minMaxLoc(result)
            return float(score), top_left, (feed_w, feed_h)

        result = cv2.matchTemplate(kill_feed_gray, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, top_left = cv2.minMaxLoc(result)
        return float(score), top_left, (tpl_w, tpl_h)

    @staticmethod
    def _crop_with_padding(
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        pad_x: int = 8,
        pad_y: int = 6,
    ) -> np.ndarray:
        img_h, img_w = image.shape[:2]
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(img_w, x + w + pad_x)
        y1 = min(img_h, y + h + pad_y)
        return image[y0:y1, x0:x1]

    @staticmethod
    def _preprocess_for_ocr(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = cv2.bitwise_not(binary)
        return binary, inverted

    def _extract_text(self, region: np.ndarray) -> str:
        if not self._ocr_available:
            return ""
        if region is None or region.size == 0:
            return ""

        binary, inverted = self._preprocess_for_ocr(region)
        ocr_config = "--oem 3 --psm 7"

        try:
            text_binary = pytesseract.image_to_string(binary, config=ocr_config)
            text_inverted = pytesseract.image_to_string(inverted, config=ocr_config)
        except pytesseract.pytesseract.TesseractNotFoundError:
            self._ocr_available = False
            if not self._warned_missing_tesseract:
                LOGGER.warning("Tesseract not found. Username-based kill detection disabled.")
                self._warned_missing_tesseract = True
            return ""
        except pytesseract.pytesseract.TesseractError:
            return ""

        return " ".join((text_binary, text_inverted)).replace("\n", " ").strip()

    def _score_username(self, text: str) -> float:
        normalized_text = self._normalize_text(text)
        if not normalized_text or not self.player_username_norm:
            return 0.0
        if self.player_username_norm in normalized_text:
            return 1.0
        full_score = self._best_substring_ratio(self.player_username_norm, normalized_text)

        token_scores: list[float] = []
        for token in self.player_username_tokens:
            if len(token) < 3:
                continue
            token_scores.append(self._best_substring_ratio(token, normalized_text))

        if not token_scores:
            return full_score

        max_token = max(token_scores)
        avg_token = float(sum(token_scores) / len(token_scores))
        token_blend = (0.65 * max_token) + (0.35 * avg_token)
        return max(full_score, token_blend)

    @staticmethod
    def _candidate_name_regions(row_crop: np.ndarray) -> dict[str, np.ndarray]:
        if row_crop is None or row_crop.size == 0:
            return {}

        height, width = row_crop.shape[:2]
        if height < 4 or width < 20:
            return {"full": row_crop}

        return {
            "full": row_crop,
            "left": row_crop[:, int(width * 0.06) : int(width * 0.46)],
            "center": row_crop[:, int(width * 0.18) : int(width * 0.70)],
            "right": row_crop[:, int(width * 0.52) : int(width * 0.96)],
        }

    def _red_background_ratio(self, row_crop: np.ndarray) -> float:
        if row_crop is None or row_crop.size == 0:
            return 1.0

        height, width = row_crop.shape[:2]
        # Exclude the left icon strip and focus on text/background area.
        background = row_crop[:, int(width * 0.18) : width]
        if background.size == 0:
            return 1.0

        hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        lower_red_1 = np.array([0, 70, 45], dtype=np.uint8)
        upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([165, 70, 45], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(
            hsv, lower_red_2, upper_red_2
        )
        return float(np.count_nonzero(mask) / mask.size)

    @staticmethod
    def _left_icon_region(row_crop: np.ndarray) -> np.ndarray:
        if row_crop is None or row_crop.size == 0:
            return row_crop
        height, width = row_crop.shape[:2]
        x0, x1 = int(width * 0.01), int(width * 0.19)
        y0, y1 = int(height * 0.08), int(height * 0.92)
        return row_crop[y0:y1, x0:x1]

    @staticmethod
    def _preprocess_icon_crop(icon_crop: np.ndarray) -> np.ndarray | None:
        if icon_crop is None or icon_crop.size == 0:
            return None
        gray = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.equalizeHist(gray)
        return gray

    def _match_killfeed_agent(self, row_crop: np.ndarray) -> tuple[str, float]:
        if not self.killfeed_agent_templates:
            return "Unknown", 0.0

        icon_crop = self._left_icon_region(row_crop)
        icon_gray = self._preprocess_icon_crop(icon_crop)
        if icon_gray is None:
            return "Unknown", 0.0

        icon_h, icon_w = icon_gray.shape[:2]
        if icon_h < 6 or icon_w < 6:
            return "Unknown", 0.0

        icon_edge = cv2.Canny(icon_gray, 60, 150)

        best_agent = "Unknown"
        best_score = -1.0
        for agent_name, template in self.killfeed_agent_templates.items():
            resized = cv2.resize(template, (icon_w, icon_h), interpolation=cv2.INTER_AREA)
            resized = cv2.equalizeHist(resized)

            result = cv2.matchTemplate(icon_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)

            edge_tpl = cv2.Canny(resized, 60, 150)
            edge_result = cv2.matchTemplate(icon_edge, edge_tpl, cv2.TM_CCOEFF_NORMED)
            _, edge_score, _, _ = cv2.minMaxLoc(edge_result)

            combined = float(max(score, edge_score * 0.92))
            if combined > best_score:
                best_score = combined
                best_agent = agent_name

        if best_score < self.killfeed_agent_threshold:
            return "Unknown", best_score
        return best_agent, best_score

    def _classify_candidate(
        self, row_crop: np.ndarray
    ) -> tuple[bool, float, dict[str, float], str, float, str, float]:
        regions = self._candidate_name_regions(row_crop)
        if not regions:
            return False, 0.0, {}, "", 1.0, "Unknown", 0.0

        scores: dict[str, float] = {}
        texts: dict[str, str] = {}
        for name, region in regions.items():
            text = self._extract_text(region)
            texts[name] = text
            scores[name] = self._score_username(text)

        left_score = scores.get("left", 0.0)
        center_score = scores.get("center", 0.0)
        full_score = scores.get("full", 0.0)
        right_score = scores.get("right", 0.0)

        # Template v2 can shift row alignment, so use strongest non-right username match.
        username_score = max(left_score, center_score, full_score)
        killer_side_match = username_score >= self.username_match_threshold
        victim_side_dominant = False

        red_ratio = self._red_background_ratio(row_crop)
        background_non_red = red_ratio <= self.non_red_max_ratio
        is_candidate = killer_side_match and not victim_side_dominant and background_non_red

        agent_name, agent_score = self._match_killfeed_agent(row_crop)
        confidence = max(username_score, right_score * 0.4)
        best_text_region = max(scores, key=scores.get) if scores else "full"
        best_text = texts.get(best_text_region, "")

        return is_candidate, confidence, scores, best_text, red_ratio, agent_name, agent_score

    def _cluster_candidates(self, candidates: list[dict]) -> list[list[dict]]:
        if not candidates:
            return []

        clusters: list[list[dict]] = [[candidates[0]]]
        for item in candidates[1:]:
            prev = clusters[-1][-1]
            time_gap = item["timestamp"] - prev["timestamp"]
            same_row_band = abs(item["row_y"] - prev["row_y"]) <= 14
            if time_gap <= self.cluster_gap_seconds and same_row_band:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        return clusters

    def detect_kills(self, sampled_frames: list[tuple[float, np.ndarray]], roi_detector) -> list[float]:
        self.last_detected_agent_name = "Unknown"
        if self.ui_template is None:
            return []

        candidates: list[dict] = []
        for timestamp, frame in sampled_frames:
            kill_feed_crop = roi_detector.crop(frame, "kill_feed")
            if kill_feed_crop is None or kill_feed_crop.size == 0:
                continue

            kill_feed_gray = self._preprocess(kill_feed_crop)
            ui_score, top_left, match_size = self._match_template_in_feed(
                kill_feed_gray, self.ui_template
            )
            if ui_score < self.ui_match_threshold:
                continue

            row_crop = self._crop_with_padding(
                kill_feed_crop,
                x=top_left[0],
                y=top_left[1],
                w=match_size[0],
                h=match_size[1],
            )

            (
                is_candidate,
                confidence,
                scores,
                matched_text,
                red_ratio,
                agent_name,
                agent_score,
            ) = self._classify_candidate(row_crop)
            if not is_candidate:
                continue

            candidates.append(
                {
                    "timestamp": float(timestamp),
                    "row_y": int(top_left[1]),
                    "confidence": float(confidence),
                    "ui_score": float(ui_score),
                    "scores": scores,
                    "text": matched_text,
                    "red_ratio": float(red_ratio),
                    "agent_name": agent_name,
                    "agent_score": float(agent_score),
                }
            )

        clusters = self._cluster_candidates(candidates)
        kill_events: list[float] = []
        agent_votes: list[str] = []
        for cluster in clusters:
            if len(cluster) < self.min_cluster_hits:
                continue
            timestamp = cluster[0]["timestamp"]
            kill_events.append(round(timestamp, 3))

            representative = max(cluster, key=lambda item: item["confidence"])
            if representative["agent_name"] != "Unknown":
                agent_votes.append(representative["agent_name"])

            LOGGER.debug(
                (
                    "Kill cluster start=%.3fs hits=%d ui=%.3f red=%.3f "
                    "left=%.3f center=%.3f right=%.3f agent=%s(%.3f) text='%s'"
                ),
                timestamp,
                len(cluster),
                representative["ui_score"],
                representative["red_ratio"],
                representative["scores"].get("left", 0.0),
                representative["scores"].get("center", 0.0),
                representative["scores"].get("right", 0.0),
                representative["agent_name"],
                representative["agent_score"],
                representative["text"][:120],
            )

        if agent_votes:
            self.last_detected_agent_name = Counter(agent_votes).most_common(1)[0][0]

        return kill_events
