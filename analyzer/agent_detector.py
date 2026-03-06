from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import AGENT_ICON_DIR, AGENT_MATCH_THRESHOLD


class AgentDetector:
    def __init__(
        self,
        icon_dir: str | Path = AGENT_ICON_DIR,
        threshold: float = AGENT_MATCH_THRESHOLD,
    ) -> None:
        self.icon_dir = Path(icon_dir)
        self.threshold = threshold
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, np.ndarray]:
        templates: dict[str, np.ndarray] = {}
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            for template_path in sorted(self.icon_dir.glob(pattern)):
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if template is None or template.size == 0:
                    continue
                templates[template_path.stem.lower()] = template
        return templates

    @staticmethod
    def _preprocess(crop) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.equalizeHist(gray)

    @staticmethod
    def _template_score(crop_gray: np.ndarray, template: np.ndarray) -> float:
        crop_h, crop_w = crop_gray.shape[:2]
        tpl_h, tpl_w = template.shape[:2]

        if crop_h < 2 or crop_w < 2:
            return 0.0

        if tpl_h > crop_h or tpl_w > crop_w:
            resized_tpl = cv2.resize(template, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(crop_gray, resized_tpl, cv2.TM_CCOEFF_NORMED)
        else:
            result = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)

        _, max_score, _, _ = cv2.minMaxLoc(result)
        return float(max_score)

    def detect_agent(self, agent_crops: list) -> str:
        if not self.templates:
            return "Unknown"

        score_history: dict[str, list[float]] = defaultdict(list)
        for crop in agent_crops:
            if crop is None or crop.size == 0:
                continue
            crop_gray = self._preprocess(crop)
            for agent_name, template in self.templates.items():
                score = self._template_score(crop_gray, template)
                score_history[agent_name].append(score)

        if not score_history:
            return "Unknown"

        final_scores: dict[str, float] = {}
        for agent_name, scores in score_history.items():
            if not scores:
                continue
            top_k = max(1, len(scores) // 4)
            top_scores = sorted(scores, reverse=True)[:top_k]
            final_scores[agent_name] = float(np.mean(top_scores))

        if not final_scores:
            return "Unknown"

        best_agent = max(final_scores, key=final_scores.get)
        if final_scores[best_agent] < self.threshold:
            return "Unknown"
        return best_agent.title()

