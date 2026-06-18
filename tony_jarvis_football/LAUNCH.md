# Launch & Troubleshooting

Quick run and troubleshooting notes for the TONY.JARVIS Gesture Football project.

## Run (conda env)

```bash
cd /home/mdnaim/MyDesk/Reinforcement_Learning/tony_jarvis_football
conda run -n tony_env python run.py
```

Note: the viewer starts in fullscreen by default.

## Common runtime notes

- The first run may download the YOLO model (`yolov8n.pt`, ~6.2MB).
- MediaPipe writes TFLite/OpenGL initialization logs — these are normal.

## Keys

- `q` or `Esc`: quit
- `r`: reset the football selection/state

## Fonts / Qt warnings (optional fix)

If you see repeated warnings like:

```
QFontDatabase: Cannot find font directory /home/.../site-packages/cv2/qt/fonts
```

Fix 1 — install system fonts (Debian/Ubuntu):

```bash
sudo apt update
sudo apt install fonts-dejavu-core
```

Fix 2 — copy fonts into the cv2 qt folder (no sudo required if you own the env):

```bash
mkdir -p /home/mdnaim/miniconda3/envs/tony_env/lib/python3.10/site-packages/cv2/qt/fonts
cp /usr/share/fonts/truetype/dejavu/*.ttf /home/mdnaim/miniconda3/envs/tony_env/lib/python3.10/site-packages/cv2/qt/fonts/
```

After installing/copying fonts, restart the app.

## Display / Wayland note

If running under Wayland or a headless session, OpenCV fullscreen behavior may vary — test on your desktop session (X11/Wayland) and ensure a physical display is available.

---

If you want a different filename, more details, or an abbreviated `README`-style summary, tell me and I’ll update the file.
