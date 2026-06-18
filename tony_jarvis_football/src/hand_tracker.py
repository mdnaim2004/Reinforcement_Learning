from dataclasses import dataclass
from typing import List

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandData:
    landmarks_xy: np.ndarray
    handedness: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]


class HandTracker:
    def __init__(self, max_num_hands: int = 2) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1,
        )

    def process(self, frame_bgr: np.ndarray) -> List[HandData]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return []

        h, w = frame_bgr.shape[:2]
        hand_data: List[HandData] = []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            pts = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pts.append([x, y])
            pts_np = np.array(pts, dtype=np.float32)

            x_min = int(np.min(pts_np[:, 0]))
            y_min = int(np.min(pts_np[:, 1]))
            x_max = int(np.max(pts_np[:, 0]))
            y_max = int(np.max(pts_np[:, 1]))

            handedness = "Unknown"
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label

            center = (float(np.mean(pts_np[:, 0])), float(np.mean(pts_np[:, 1])))
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
        self._hands.close()
