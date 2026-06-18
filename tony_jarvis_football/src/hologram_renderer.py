from dataclasses import dataclass, field
import math
import random
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float


@dataclass
class BallVisualState:
    center: np.ndarray
    radius: float
    rotation_deg: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    selected: bool = False
    grabbed: bool = False


class HologramRenderer:
    def __init__(self) -> None:
        self.particles: List[Particle] = []

    def spawn_particles(self, center: Tuple[float, float], count: int = 8) -> None:
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(0.5, 2.8)
            self.particles.append(
                Particle(
                    x=center[0],
                    y=center[1],
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    life=random.uniform(12, 28),
                )
            )

    def update_particles(self) -> None:
        survivors: List[Particle] = []
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.life -= 1.0
            if p.life > 0:
                survivors.append(p)
        self.particles = survivors

    def draw_hologram_ball(self, frame: np.ndarray, ball: BallVisualState) -> np.ndarray:
        overlay = frame.copy()
        center = (int(ball.center[0]), int(ball.center[1]))
        radius = int(ball.radius)

        cv2.circle(overlay, center, radius, (255, 180, 40), 2, lineType=cv2.LINE_AA)
        cv2.circle(overlay, center, int(radius * 0.78), (255, 130, 20), 1, lineType=cv2.LINE_AA)

        for idx, ratio in enumerate((1.15, 1.35, 1.58)):
            ring_radius = int(radius * ratio)
            start_angle = int((ball.rotation_deg + idx * 40) % 360)
            end_angle = int((start_angle + 220) % 360)
            cv2.ellipse(
                overlay,
                center,
                (ring_radius, int(ring_radius * 0.45)),
                (ball.rotation_deg + idx * 15) % 360,
                start_angle,
                end_angle,
                (255, 210, 120),
                1,
                lineType=cv2.LINE_AA,
            )

        for spoke in range(0, 360, 45):
            theta = math.radians(spoke + ball.rotation_deg)
            x1 = int(center[0] + math.cos(theta) * radius * 1.05)
            y1 = int(center[1] + math.sin(theta) * radius * 1.05)
            x2 = int(center[0] + math.cos(theta) * radius * 1.32)
            y2 = int(center[1] + math.sin(theta) * radius * 1.32)
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 200, 100), 1, lineType=cv2.LINE_AA)

        if ball.selected:
            cv2.circle(overlay, center, int(radius * 1.7), (255, 230, 150), 1, lineType=cv2.LINE_AA)
        if ball.grabbed:
            cv2.circle(overlay, center, int(radius * 0.6), (255, 240, 180), 2, lineType=cv2.LINE_AA)

        self.update_particles()
        for p in self.particles:
            alpha = min(max(p.life / 30.0, 0.0), 1.0)
            color = (int(230 * alpha), int(180 * alpha), int(90 * alpha))
            cv2.circle(overlay, (int(p.x), int(p.y)), 1, color, -1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return frame

    def draw_hud(self, frame: np.ndarray, labels: tuple[str, ...], fps: float) -> np.ndarray:
        overlay = frame.copy()
        h, w = frame.shape[:2]

        cv2.rectangle(overlay, (20, 20), (420, 160), (140, 60, 10), -1)
        cv2.putText(
            overlay,
            "JARVIS Football Control",
            (35, 52),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 220, 120),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"FPS: {fps:5.1f}",
            (35, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 220, 120),
            1,
            lineType=cv2.LINE_AA,
        )

        y = 112
        if labels:
            label_text = " | ".join(labels[:3])
            cv2.putText(
                overlay,
                label_text,
                (35, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 190, 80),
                1,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.putText(
                overlay,
                "Awaiting gesture input",
                (35, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 190, 80),
                1,
                lineType=cv2.LINE_AA,
            )

        cv2.line(overlay, (w - 170, 35), (w - 35, 35), (200, 120, 20), 1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (w - 35, 35), (w - 35, 170), (200, 120, 20), 1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (35, h - 170), (170, h - 170), (200, 120, 20), 1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (35, h - 170), (35, h - 35), (200, 120, 20), 1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
        return frame
