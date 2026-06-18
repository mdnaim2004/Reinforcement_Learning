from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import urllib.request

import cv2
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


HAND_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"


@dataclass
class HandData:
    landmarks_xy: np.ndarray
    handedness: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]


class HandTracker:
    def __init__(self, max_num_hands: int = 2) -> None:
        self._landmarker: Optional[mp_vision.HandLandmarker] = None
        self._model_path = self._ensure_model_asset()

        if self._model_path is None:
            return

        base_options = mp_tasks.BaseOptions(
            model_asset_path=str(self._model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(
            options)

    @staticmethod
    def _asset_path() -> Path:
        return Path(__file__).resolve().parent.parent / "assets" / "hand_landmarker.task"

    def _ensure_model_asset(self) -> Optional[Path]:
        asset_path = self._asset_path()
        if asset_path.exists() and asset_path.stat().st_size > 0:
            return asset_path

        asset_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_URL, asset_path)
        except Exception:
            if asset_path.exists():
                asset_path.unlink(missing_ok=True)
            return None

        return asset_path if asset_path.exists() and asset_path.stat().st_size > 0 else None

    def process(self, frame_bgr: np.ndarray) -> List[HandData]:
        if self._landmarker is None:
            return []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp_vision.RunningMode.IMAGE
        image = mp_vision.Image(
            image_format=mp_vision.ImageFormat.SRGB, data=frame_rgb)
        results = self._landmarker.detect(image)

        if not results.hand_landmarks:
            return []

        h, w = frame_bgr.shape[:2]
        hand_data: List[HandData] = []

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            pts = []
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pts.append([x, y])
            pts_np = np.array(pts, dtype=np.float32)

            x_min = int(np.min(pts_np[:, 0]))
            y_min = int(np.min(pts_np[:, 1]))
            x_max = int(np.max(pts_np[:, 0]))
            y_max = int(np.max(pts_np[:, 1]))

            handedness = "Unknown"
            if results.handedness and idx < len(results.handedness):
                handedness = results.handedness[idx][0].category_name

            center = (float(np.mean(pts_np[:, 0])),
                      float(np.mean(pts_np[:, 1])))
            hand_data.append(
                HandData(
                    landmarks_xy=pts_np,
                    handedness=handedness,
                    bbox=(x_min, y_min, x_max, y_max),
                    center=center,
                )
            )

        return hand_data

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
