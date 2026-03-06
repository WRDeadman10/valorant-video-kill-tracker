from __future__ import annotations

from dataclasses import dataclass

from config import ROI_CONFIG, NormalizedROI


@dataclass(frozen=True)
class PixelROI:
    x: int
    y: int
    width: int
    height: int


class ROIDetector:
    def __init__(self, roi_config: dict[str, NormalizedROI] | None = None) -> None:
        self._roi_config = roi_config or ROI_CONFIG

    @staticmethod
    def _to_pixel_roi(norm_roi: NormalizedROI, frame_shape: tuple[int, ...]) -> PixelROI:
        frame_height, frame_width = frame_shape[:2]

        x = max(0, int(norm_roi.x * frame_width))
        y = max(0, int(norm_roi.y * frame_height))
        width = max(1, int(norm_roi.width * frame_width))
        height = max(1, int(norm_roi.height * frame_height))

        if x + width > frame_width:
            width = frame_width - x
        if y + height > frame_height:
            height = frame_height - y

        return PixelROI(x=x, y=y, width=width, height=height)

    def get_pixel_roi(self, frame, region_name: str) -> PixelROI:
        if region_name not in self._roi_config:
            raise KeyError(f"Unknown ROI region: {region_name}")
        return self._to_pixel_roi(self._roi_config[region_name], frame.shape)

    def crop(self, frame, region_name: str):
        roi = self.get_pixel_roi(frame, region_name)
        return frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

