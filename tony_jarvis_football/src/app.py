import time
from typing import Optional

import cv2
import numpy as np

from .config import SimulationConfig
from .gesture_engine import GestureEngine
from .hand_tracker import HandTracker
from .hologram_renderer import BallVisualState, HologramRenderer


try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class FootballSimulation:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.tracker = HandTracker(max_num_hands=config.hand_max_count)
        self.gesture_engine = GestureEngine()
        self.renderer = HologramRenderer()

        self.ball = BallVisualState(
            center=np.array([640.0, 360.0], dtype=np.float32),
            radius=config.base_ball_radius,
            rotation_deg=0.0,
        )

        self.yolo_model: Optional[YOLO] = None
        if YOLO is not None:
            try:
                self.yolo_model = YOLO(config.yolo_model_name)
            except Exception:
                self.yolo_model = None

        self.last_frame_time = time.time()

    @staticmethod
    def _create_fullscreen_window(window_name: str) -> None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def _estimate_ball_from_yolo(self, frame: np.ndarray) -> None:
        if self.yolo_model is None:
            return

        try:
            results = self.yolo_model.predict(
                frame,
                classes=[self.config.yolo_ball_class_id],
                conf=0.35,
                verbose=False,
                max_det=1,
            )
        except Exception:
            return

        if not results:
            return

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return

        box = boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        radius = max((x2 - x1), (y2 - y1)) * 0.45

        self.ball.center = np.array([cx, cy], dtype=np.float32)
        self.ball.radius = float(
            np.clip(radius, self.config.min_ball_radius, self.config.max_ball_radius))

    def _apply_gestures(self, gesture_output, hands, frame_shape) -> None:
        h, w = frame_shape[:2]

        if gesture_output.select:
            self.ball.selected = True

        if gesture_output.grab and self.ball.selected:
            self.ball.grabbed = True
        if gesture_output.release:
            self.ball.grabbed = False

        self.ball.radius += gesture_output.scale_delta * self.config.grow_shrink_speed
        self.ball.radius = float(np.clip(
            self.ball.radius, self.config.min_ball_radius, self.config.max_ball_radius))

        self.ball.rotation_deg = (
            self.ball.rotation_deg + gesture_output.rotate_delta * self.config.rotate_speed) % 360

        if self.ball.grabbed and hands:
            self.ball.center = np.array(hands[0].center, dtype=np.float32)
            self.ball.velocity *= 0.0
        else:
            self.ball.velocity[1] += self.config.gravity
            self.ball.center += self.ball.velocity

            if self.ball.center[0] <= self.ball.radius:
                self.ball.center[0] = self.ball.radius
                self.ball.velocity[0] *= -self.config.restitution
            if self.ball.center[0] >= (w - self.ball.radius):
                self.ball.center[0] = w - self.ball.radius
                self.ball.velocity[0] *= -self.config.restitution
            if self.ball.center[1] <= self.ball.radius:
                self.ball.center[1] = self.ball.radius
                self.ball.velocity[1] *= -self.config.restitution
            if self.ball.center[1] >= (h - self.ball.radius):
                self.ball.center[1] = h - self.ball.radius
                self.ball.velocity[1] *= -self.config.restitution

        if gesture_output.launch and self.ball.selected:
            self.ball.velocity = np.array(
                [0.0, self.config.launch_velocity], dtype=np.float32)
            self.renderer.spawn_particles(
                (float(self.ball.center[0]), float(self.ball.center[1])), count=24)

    def run(self) -> None:
        window_name = "TONY.JARVIS Gesture Football"
        self._create_fullscreen_window(window_name)
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Unable to open webcam")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)

                if not self.ball.selected:
                    self._estimate_ball_from_yolo(frame)

                hands = self.tracker.process(frame)
                gesture_output = self.gesture_engine.process(
                    hands, frame.shape)

                self._apply_gestures(gesture_output, hands, frame.shape)
                if self.ball.selected:
                    self.renderer.spawn_particles(
                        (float(self.ball.center[0]), float(self.ball.center[1])), count=1)

                frame = self.renderer.draw_hologram_ball(frame, self.ball)

                now = time.time()
                fps = 1.0 / max(now - self.last_frame_time, 1e-4)
                self.last_frame_time = now

                frame = self.renderer.draw_hud(
                    frame, gesture_output.active_labels, fps)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    self.ball.selected = False
                    self.ball.grabbed = False
                    self.ball.velocity[:] = 0.0

        finally:
            self.tracker.close()
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    sim = FootballSimulation(SimulationConfig())
    sim.run()
