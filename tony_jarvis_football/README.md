# TONY JARVIS Gesture Football Simulation

A JARVIS-like, gesture-controlled football simulation with a holographic sci-fi HUD.

## Stack

- YOLO (sports ball detection)
- MediaPipe (hand tracking)
- OpenCV (camera + rendering)
- NumPy (math + simulation state)

## Architecture

camera -> hand tracking -> gesture recognition -> holographic rendering -> UI overlay

## Gesture Map

- Hands apart -> football grows
- Hands together -> football shrinks
- Closed fist -> grab football
- Open palm -> release football
- Swipe left/right -> rotate football
- Raise both hands -> launch football animation
- Pinch gesture -> select football

## Visual Style

- Blue hologram football shell
- Transparent energy sphere layers
- Circular HUD rings
- Particle system around football
- Futuristic sci-fi interface
- Tony Stark inspired overlay lines/panels

## Quick Start

```bash
cd tony_jarvis_football
bash scripts/create_tony_environment.sh
source tony_env/bin/activate
python run.py
```

Press `q` (or `Esc`) to quit.
Press `r` to reset football selection and control.

## Notes

- If YOLO model download fails or `ultralytics` is unavailable, the simulation still runs with gesture-only control.
- Keep your hand and full upper body in frame for better swipe and both-hands-raised detection.
