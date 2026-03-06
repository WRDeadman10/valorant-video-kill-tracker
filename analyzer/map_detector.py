from __future__ import annotations

from collections import Counter
import difflib
import logging

import cv2
import pytesseract

from config import KNOWN_MAPS


LOGGER = logging.getLogger(__name__)


class MapDetector:
    def __init__(self, known_maps: list[str] | None = None) -> None:
        self.known_maps = known_maps or KNOWN_MAPS
        self._normalized_map_lookup = {self._normalize_text(name): name for name in self.known_maps}

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(ch for ch in text.upper() if ch.isalpha())

    @staticmethod
    def _preprocess_for_ocr(map_crop):
        gray = cv2.cvtColor(map_crop, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        filtered = cv2.GaussianBlur(enlarged, (3, 3), 0)
        _, thresholded = cv2.threshold(
            filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresholded

    def _match_map(self, ocr_text: str) -> str | None:
        cleaned = self._normalize_text(ocr_text)
        if not cleaned:
            return None

        for normalized, original in self._normalized_map_lookup.items():
            if normalized in cleaned or cleaned in normalized:
                return original

        close = difflib.get_close_matches(
            cleaned, self._normalized_map_lookup.keys(), n=1, cutoff=0.60
        )
        if close:
            return self._normalized_map_lookup[close[0]]
        return None

    def detect_map(self, map_crops: list) -> str:
        votes = Counter()
        tesseract_available = True
        for crop in map_crops:
            if crop is None or crop.size == 0:
                continue
            processed = self._preprocess_for_ocr(crop)
            if not tesseract_available:
                break
            try:
                ocr_text = pytesseract.image_to_string(processed, config="--psm 7 --oem 3")
            except pytesseract.pytesseract.TesseractNotFoundError:
                LOGGER.warning("Tesseract executable not found. Map detection set to Unknown.")
                tesseract_available = False
                break
            except pytesseract.pytesseract.TesseractError:
                continue
            detected = self._match_map(ocr_text)
            if detected:
                votes[detected] += 1

        if not votes:
            return "Unknown"
        return votes.most_common(1)[0][0]
