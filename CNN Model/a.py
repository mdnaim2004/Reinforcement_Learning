"""
╔══════════════════════════════════════════════════════════════╗
║          HAND GESTURE CONTROLLER - OpenCV + MediaPipe        ║
║  Features: Air Write | Erase | Mouse Control                 ║
╚══════════════════════════════════════════════════════════════╝

GESTURE GUIDE:
─────────────────────────────────────────────────────────────
  ✍️  AIR WRITE      →  Index finger up only
  🖱️  MOUSE CONTROL  →  Index + Middle finger up (together)
  🧹  ERASE (stroke) →  Index + Middle + Ring finger up
  🗑️  CLEAR ALL      →  All 5 fingers up (open palm)
  ✊  PAUSE / IDLE   →  Fist (all fingers down)
─────────────────────────────────────────────────────────────

CONTROLS:
  [Q] or [ESC]  →  Quit
  [C]           →  Clear canvas
  [S]           →  Save canvas as PNG
  [1-4]         →  Change pen color
  [+/-]         →  Change brush size
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import os
from collections import deque

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CAM_WIDTH       = 1280
CAM_HEIGHT      = 720
SMOOTHING       = 7          # Mouse smoothing frames
DRAW_SMOOTHING  = 3          # Drawing smoothing frames
ERASER_SIZE     = 50
MIN_DETECTION   = 0.8
MIN_TRACKING    = 0.8

COLORS = {
    "WHITE"  : (255, 255, 255),
    "CYAN"   : (255, 255,   0),
    "GREEN"  : ( 57, 255,  20),
    "PINK"   : (255,  20, 147),
    "YELLOW" : (  0, 255, 255),
    "RED"    : ( 50,  50, 255),
}

PEN_COLORS = [
    (255, 255, 255),   # 1 - White
    (  0, 255, 255),   # 2 - Yellow
    ( 57, 255,  20),   # 3 - Green
    (255,  50,  50),   # 4 - Blue
]

COLOR_NAMES = ["White", "Yellow", "Green", "Blue"]

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION,
    min_tracking_confidence=MIN_TRACKING
)

# ─────────────────────────────────────────────
#  PYAUTOGUI SETUP
# ─────────────────────────────────────────────
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()


# ─────────────────────────────────────────────
#  FINGER DETECTION
# ─────────────────────────────────────────────
def get_fingers_up(landmarks, handedness="Right"):
    """Returns list of booleans: [thumb, index, middle, ring, pinky]"""
    tips   = [4, 8, 12, 16, 20]
    joints = [3, 6, 10, 14, 18]
    fingers = []

    # Thumb (different axis check)
    if handedness == "Right":
        fingers.append(landmarks[tips[0]].x < landmarks[joints[0]].x)
    else:
        fingers.append(landmarks[tips[0]].x > landmarks[joints[0]].x)

    # Other 4 fingers (y-axis)
    for i in range(1, 5):
        fingers.append(landmarks[tips[i]].y < landmarks[joints[i]].y)

    return fingers


def get_gesture(fingers):
    """Map finger states to gesture name"""
    thumb, index, middle, ring, pinky = fingers

    if not any(fingers):
        return "IDLE"                                # fist
    if all(fingers):
        return "CLEAR"                               # open palm
    if index and not middle and not ring and not pinky:
        return "WRITE"                               # index only
    if index and middle and not ring and not pinky:
        return "MOUSE"                               # peace sign
    if index and middle and ring and not pinky:
        return "ERASE"                               # 3 fingers
    return "UNKNOWN"


# ─────────────────────────────────────────────
#  DRAWING OVERLAY
# ─────────────────────────────────────────────
def draw_ui(frame, canvas, mode, color_idx, brush_size,
            fps, mouse_pos, prev_gesture, gesture_timer):
    """Draw the heads-up display on frame"""
    h, w = frame.shape[:2]

    # Blend canvas onto frame
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 5, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frame = np.where(mask_3ch > 0, canvas, frame)

    # ── Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    # ── Mode badge
    mode_colors = {
        "WRITE": (57, 255, 20),
        "MOUSE": (0, 200, 255),
        "ERASE": (0, 100, 255),
        "CLEAR": (0, 50, 200),
        "IDLE":  (100, 100, 100),
        "UNKNOWN": (80, 80, 80),
    }
    mc = mode_colors.get(mode, (150, 150, 150))
    mode_icons = {
        "WRITE": "✍ WRITE",
        "MOUSE": "🖱 MOUSE",
        "ERASE": "🧹 ERASE",
        "CLEAR": "🗑 CLEAR",
        "IDLE":  "✊ IDLE",
        "UNKNOWN": "❓",
    }
    label = mode_icons.get(mode, mode)
    cv2.rectangle(frame, (10, 8), (175, 55), mc, -1)
    cv2.rectangle(frame, (10, 8), (175, 55), (255,255,255), 1)
    cv2.putText(frame, label, (18, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)

    # ── Color swatch
    pen_color = PEN_COLORS[color_idx]
    cv2.rectangle(frame, (190, 10), (230, 53), pen_color, -1)
    cv2.rectangle(frame, (190, 10), (230, 53), (255,255,255), 1)
    cv2.putText(frame, COLOR_NAMES[color_idx], (235, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)

    # ── Brush size
    cv2.putText(frame, f"Size: {brush_size}", (340, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.circle(frame, (445, 32), brush_size // 2, pen_color, -1)

    # ── FPS
    cv2.putText(frame, f"FPS: {fps:02d}", (w - 110, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,150), 1)

    # ── Bottom help bar
    help_y = h - 12
    cv2.rectangle(frame, (0, h-30), (w, h), (15,15,15), -1)
    hints = "[Q] Quit  [C] Clear  [S] Save  [1-4] Color  [+/-] Size"
    cv2.putText(frame, hints, (10, help_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    return frame


def draw_cursor(frame, cx, cy, gesture, color, brush_size):
    """Draw animated cursor at fingertip"""
    if gesture == "WRITE":
        cv2.circle(frame, (cx, cy), brush_size // 2 + 2, (255,255,255), 2)
        cv2.circle(frame, (cx, cy), brush_size // 2, color, -1)
    elif gesture == "ERASE":
        cv2.rectangle(frame,
                      (cx - ERASER_SIZE, cy - ERASER_SIZE),
                      (cx + ERASER_SIZE, cy + ERASER_SIZE),
                      (0, 0, 200), 2)
        cv2.putText(frame, "ERASE", (cx - 22, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    elif gesture == "MOUSE":
        cv2.circle(frame, (cx, cy), 10, (0,200,255), 2)
        cv2.line(frame, (cx-6, cy), (cx+6, cy), (0,200,255), 2)
        cv2.line(frame, (cx, cy-6), (cx, cy+6), (0,200,255), 2)


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Drawing canvas (black)
    canvas = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)

    # State
    draw_points    = deque(maxlen=DRAW_SMOOTHING)
    mouse_points   = deque(maxlen=SMOOTHING)
    prev_draw_pt   = None
    color_idx      = 0
    brush_size     = 8
    prev_gesture   = "IDLE"
    gesture_timer  = 0

    fps_counter = deque(maxlen=30)
    save_count  = 0

    print(__doc__)
    print("🟢 Camera started. Show your hand to begin!\n")

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error!"); break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture  = "IDLE"
        cx, cy   = -1, -1
        fingers  = [False]*5

        if result.multi_hand_landmarks and result.multi_handedness:
            hand_lm   = result.multi_hand_landmarks[0]
            handedness = result.multi_handedness[0].classification[0].label

            # Draw skeleton
            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            lm = hand_lm.landmark
            fingers  = get_fingers_up(lm, handedness)
            gesture  = get_gesture(fingers)

            # Index fingertip position
            cx = int(lm[8].x * w)
            cy = int(lm[8].y * h)

            # ── WRITE MODE
            if gesture == "WRITE":
                draw_points.append((cx, cy))
                avg_x = int(np.mean([p[0] for p in draw_points]))
                avg_y = int(np.mean([p[1] for p in draw_points]))

                if prev_draw_pt and prev_gesture == "WRITE":
                    cv2.line(canvas, prev_draw_pt, (avg_x, avg_y),
                             PEN_COLORS[color_idx], brush_size)
                prev_draw_pt = (avg_x, avg_y)

            else:
                prev_draw_pt = None
                draw_points.clear()

            # ── MOUSE CONTROL MODE
            if gesture == "MOUSE":
                # Map camera coords to screen
                raw_mx = int(np.interp(cx, [0, w], [0, SCREEN_W]))
                raw_my = int(np.interp(cy, [0, h], [0, SCREEN_H]))
                mouse_points.append((raw_mx, raw_my))

                sm_x = int(np.mean([p[0] for p in mouse_points]))
                sm_y = int(np.mean([p[1] for p in mouse_points]))
                pyautogui.moveTo(sm_x, sm_y, _pause=False)

            else:
                mouse_points.clear()

            # ── ERASE MODE (stroke eraser)
            if gesture == "ERASE":
                cv2.rectangle(canvas,
                              (cx - ERASER_SIZE, cy - ERASER_SIZE),
                              (cx + ERASER_SIZE, cy + ERASER_SIZE),
                              (0, 0, 0), -1)

            # ── CLEAR ALL
            if gesture == "CLEAR" and prev_gesture != "CLEAR":
                canvas[:] = 0
                print("🗑  Canvas cleared!")

            # Draw cursor
            draw_cursor(frame, cx, cy, gesture, PEN_COLORS[color_idx], brush_size)

        # ── Gesture transition feedback
        if gesture != prev_gesture:
            gesture_timer = time.time()
        prev_gesture = gesture

        # ── FPS
        fps_counter.append(time.time() - t_start)
        fps = int(1.0 / (np.mean(fps_counter) + 1e-9))

        # ── Draw UI
        frame = draw_ui(frame, canvas, gesture, color_idx,
                        brush_size, fps, (cx, cy), prev_gesture, gesture_timer)

        cv2.imshow("Hand Gesture Controller", frame)

        # ── Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            print("👋 Exiting..."); break
        elif key == ord('c'):
            canvas[:] = 0
            print("🗑  Canvas cleared!")
        elif key == ord('s'):
            save_count += 1
            fname = f"gesture_canvas_{save_count:03d}.png"
            cv2.imwrite(fname, canvas)
            print(f"💾 Saved: {fname}")
        elif key == ord('1'): color_idx = 0; print(f"🎨 Color: {COLOR_NAMES[0]}")
        elif key == ord('2'): color_idx = 1; print(f"🎨 Color: {COLOR_NAMES[1]}")
        elif key == ord('3'): color_idx = 2; print(f"🎨 Color: {COLOR_NAMES[2]}")
        elif key == ord('4'): color_idx = 3; print(f"🎨 Color: {COLOR_NAMES[3]}")
        elif key == ord('+') or key == ord('='): brush_size = min(40, brush_size + 2)
        elif key == ord('-'): brush_size = max(2, brush_size - 2)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done!")


if __name__ == "__main__":
    main()
