from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .hand_tracker import HandData


@dataclass
class GestureOutput:
    scale_delta: float = 0.0
    grab: bool = False
    release: bool = False
    rotate_delta: float = 0.0
    launch: bool = False
    select: bool = False
    active_labels: tuple[str, ...] = ()


class GestureEngine:
    FINGER_TIPS = (4, 8, 12, 16, 20)
    FINGER_MCP = (2, 5, 9, 13, 17)

    def __init__(self) -> None:
        self._two_hand_distance = None
        self._center_x_history = deque(maxlen=5)
        self._launch_cooldown = 0

    @staticmethod
    def _palm_scale(hand: HandData) -> float:
        wrist = hand.landmarks_xy[0]
        middle_mcp = hand.landmarks_xy[9]
        return float(np.linalg.norm(wrist - middle_mcp) + 1e-6)

    def _is_closed_fist(self, hand: HandData) -> bool:
        palm = self._palm_scale(hand)
        tip_to_mcp = []
        for tip_idx, mcp_idx in zip(self.FINGER_TIPS, self.FINGER_MCP):
            tip_to_mcp.append(np.linalg.norm(hand.landmarks_xy[tip_idx] - hand.landmarks_xy[mcp_idx]) / palm)
        return float(np.mean(tip_to_mcp)) < 0.95

    def _is_open_palm(self, hand: HandData) -> bool:
        palm = self._palm_scale(hand)
        tip_to_wrist = []
        wrist = hand.landmarks_xy[0]
        for tip_idx in self.FINGER_TIPS:
            tip_to_wrist.append(np.linalg.norm(hand.landmarks_xy[tip_idx] - wrist) / palm)
        return float(np.mean(tip_to_wrist)) > 2.25

    def _is_pinch(self, hand: HandData) -> bool:
        palm = self._palm_scale(hand)
        thumb_tip = hand.landmarks_xy[4]
        index_tip = hand.landmarks_xy[8]
        return float(np.linalg.norm(thumb_tip - index_tip) / palm) < 0.65

    def process(self, hands: List[HandData], frame_shape: tuple[int, int, int]) -> GestureOutput:
        labels: List[str] = []
        output = GestureOutput()

        if self._launch_cooldown > 0:
            self._launch_cooldown -= 1

        if len(hands) == 2:
            dist = float(np.linalg.norm(np.array(hands[0].center) - np.array(hands[1].center)))
            if self._two_hand_distance is not None:
                delta = dist - self._two_hand_distance
                if delta > 8:
                    output.scale_delta = 1.0
                    labels.append("Hands apart")
                elif delta < -8:
                    output.scale_delta = -1.0
                    labels.append("Hands together")
            self._two_hand_distance = dist
        else:
            self._two_hand_distance = None

        if hands:
            leader = hands[0]
            self._center_x_history.append(leader.center[0])

            if self._is_closed_fist(leader):
                output.grab = True
                labels.append("Closed fist")
            if self._is_open_palm(leader):
                output.release = True
                labels.append("Open palm")
            if self._is_pinch(leader):
                output.select = True
                labels.append("Pinch")

            if len(self._center_x_history) == self._center_x_history.maxlen:
                swipe_delta = self._center_x_history[-1] - self._center_x_history[0]
                if swipe_delta > 45:
                    output.rotate_delta = 1.0
                    labels.append("Swipe right")
                elif swipe_delta < -45:
                    output.rotate_delta = -1.0
                    labels.append("Swipe left")

        if len(hands) == 2 and self._launch_cooldown == 0:
            frame_h = frame_shape[0]
            left_wrist_y = hands[0].landmarks_xy[0][1]
            right_wrist_y = hands[1].landmarks_xy[0][1]
            if left_wrist_y < frame_h * 0.35 and right_wrist_y < frame_h * 0.35:
                output.launch = True
                self._launch_cooldown = 30
                labels.append("Both hands up")

        output.active_labels = tuple(labels)
        return output
