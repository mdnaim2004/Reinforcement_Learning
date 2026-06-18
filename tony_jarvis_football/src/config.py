from dataclasses import dataclass


@dataclass
class VisualConfig:
    primary_blue: tuple[int, int, int] = (255, 180, 30)
    bright_blue: tuple[int, int, int] = (255, 230, 120)
    dark_blue: tuple[int, int, int] = (160, 80, 10)
    hud_alpha: float = 0.25
    energy_alpha: float = 0.32


@dataclass
class SimulationConfig:
    camera_index: int = 0
    hand_max_count: int = 2
    base_ball_radius: float = 55.0
    min_ball_radius: float = 25.0
    max_ball_radius: float = 160.0
    grow_shrink_speed: float = 0.9
    rotate_speed: float = 2.4
    launch_velocity: float = -13.0
    gravity: float = 0.55
    restitution: float = 0.75
    yolo_model_name: str = "yolov8n.pt"
    yolo_ball_class_id: int = 32
