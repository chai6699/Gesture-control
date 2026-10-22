"""
Hand Gesture Computer Control  v2
---------------------------------
Control your computer with hand gestures via webcam.

Gestures (palm facing camera):
  Index finger up only              -> move cursor
  Pinch thumb + index (quick)       -> left click
  Pinch thumb + index (hold)        -> drag
  Pinch thumb + middle              -> right click
  Index + middle up, move up/down   -> scroll
  Pinch thumb + ring, move up/down  -> volume
  Open palm, fast swipe left/right  -> switch desktop
  Fist                              -> pause

Press 'q' in the preview window to quit.
Safety: flick the real mouse into a screen corner to stop (PyAutoGUI failsafe).
"""

import math
import subprocess
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import pyautogui

# ---------------- Settings ----------------
CAM_INDEX = 0
CAM_W, CAM_H = 960, 540      # higher resolution = better detection at a distance
FRAME_MARGIN = 150           # px border; smaller hand movements cover the whole screen

# Tracking quality
DETECTION_CONFIDENCE = 0.75  # how sure the model must be before it locks onto a hand
TRACKING_CONFIDENCE = 0.75   # how sure it must stay; drops back to re-detecting below this
LANDMARK_SMOOTHING = 0.5     # 0 = off, closer to 1 = steadier joints but more lag
LOW_LIGHT_BOOST = True       # brighten dark frames before detection
PREFERRED_HAND = None        # "Right", "Left", or None to accept either

# Cursor smoothing (One Euro filter)
FILTER_MIN_CUTOFF = 1.0      # lower = steadier when moving slowly
FILTER_BETA = 8.0            # higher = less lag when moving fast

# Pinch (relative to hand size, so distance from camera doesn't matter)
# Per finger, because the thumb reaches the middle and ring fingers less easily
PINCH_ON = {"index": 0.35, "middle": 0.40, "ring": 0.45}   # pinch starts below this
PINCH_OFF = {"index": 0.50, "middle": 0.55, "ring": 0.60}  # pinch ends above this

# Finger straightness (1.0 = fully straight, ~0.3 = curled)
FINGER_ON = 0.85             # finger counts as up above this
FINGER_OFF = 0.75            # and stays up until it drops below this
SHOW_DEBUG = True            # show finger states on the preview window

DRAG_HOLD_TIME = 0.35       # seconds of pinch before it becomes a drag
CLICK_COOLDOWN = 0.4
SCROLL_SENSITIVITY = 40
VOLUME_SENSITIVITY = 150     # volume % per full-frame-height of movement

SWIPE_WINDOW = 0.25          # seconds to look back for a swipe
SWIPE_DISTANCE = 0.25        # fraction of frame width the palm must travel
SWIPE_COOLDOWN = 1.0

QUIT_FIST_SECONDS = 3.0      # hold a fist this long to quit
# ------------------------------------------

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()
IS_MAC = sys.platform == "darwin"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}


# ---------- Smoothing ----------
class OneEuroFilter:
    """Steady at low speed, responsive at high speed."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev, self.dx_prev = x, t, 0.0
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class LandmarkSmoother:
    """Smooths all 21 hand joints over time so finger states stop flickering."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.prev = None

    def reset(self):
        self.prev = None

    def __call__(self, lm):
        if self.alpha <= 0:
            return
        if self.prev is None:
            self.prev = [(p.x, p.y, p.z) for p in lm]
            return
        a = self.alpha
        new = []
        for p, (px, py, pz) in zip(lm, self.prev):
            p.x = a * px + (1 - a) * p.x
            p.y = a * py + (1 - a) * p.y
            p.z = a * pz + (1 - a) * p.z
            new.append((p.x, p.y, p.z))
        self.prev = new


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance(frame):
    """Brighten and add contrast only when the frame is dark."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() > 90:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ---------- System actions ----------
def get_volume():
    if IS_MAC:
        try:
            out = subprocess.check_output(
                ["osascript", "-e", "output volume of (get volume settings)"])
            return int(out.strip())
        except Exception:
            return 50
    return 50


def set_volume(v):
    v = max(0, min(100, int(v)))
    if IS_MAC:
        subprocess.Popen(["osascript", "-e", f"set volume output volume {v}"])
    return v


def nudge_volume_keys(direction):
    """Non-mac fallback: press media keys."""
    pyautogui.press("volumeup" if direction > 0 else "volumedown")


def switch_desktop(direction):
    if IS_MAC:
        pyautogui.hotkey("ctrl", "right" if direction > 0 else "left")
    else:
        pyautogui.hotkey("ctrl", "win", "right" if direction > 0 else "left")


# ---------- Hand helpers ----------
MCP = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
DIP = {"index": 7, "middle": 11, "ring": 15, "pinky": 19}
_finger_state = {f: False for f in ("index", "middle", "ring", "pinky")}
last_straightness = {}


def straightness(lm, f):
    """
    Knuckle-to-tip straight-line distance divided by the length along the
    finger's bones. 1.0 = perfectly straight, ~0.3 = fully curled.
    Works the same for long and short fingers (so the pinky is treated fairly).
    """
    m, p, d, t = lm[MCP[f]], lm[PIP[f]], lm[DIP[f]], lm[TIP[f]]
    path = dist(m, p) + dist(p, d) + dist(d, t)
    return dist(m, t) / max(path, 1e-6)


def fingers_up(lm, handedness):
    up = {}
    if handedness == "Right":
        up["thumb"] = lm[4].x < lm[3].x
    else:
        up["thumb"] = lm[4].x > lm[3].x
    for f in ("index", "middle", "ring", "pinky"):
        s = straightness(lm, f)
        last_straightness[f] = s
        if _finger_state[f]:
            _finger_state[f] = s > FINGER_OFF
        else:
            _finger_state[f] = s > FINGER_ON
        up[f] = _finger_state[f]
    return up


ASPECT = CAM_W / CAM_H


def dist(a, b):
    # x is scaled so distances aren't stretched by the widescreen frame
    return math.hypot((a.x - b.x) * ASPECT, a.y - b.y)


def hand_scale(lm):
    """Wrist to middle-finger knuckle: a size reference for the hand."""
    return max(dist(lm[0], lm[9]), 1e-6)


def map_to_screen(x, y):
    px, py = x * CAM_W, y * CAM_H
    sx = (px - FRAME_MARGIN) / (CAM_W - 2 * FRAME_MARGIN) * SCREEN_W
    sy = (py - FRAME_MARGIN) / (CAM_H - 2 * FRAME_MARGIN) * SCREEN_H
    return max(0, min(SCREEN_W - 1, sx)), max(0, min(SCREEN_H - 1, sy))


class PinchSelector:
    """
    Decides which single finger (if any) the thumb is pinching.
    Only the finger closest to the thumb can win, so pinching the middle
    finger is never mistaken for an index pinch.
    """

    def __init__(self):
        self.active = None
        self.ratios = {}

    def update(self, lm):
        scale = hand_scale(lm)
        self.ratios = {f: dist(lm[4], lm[TIP[f]]) / scale for f in ("index", "middle", "ring")}
        closest = min(self.ratios, key=self.ratios.get)
        if self.active is not None and self.ratios[self.active] < PINCH_OFF[self.active]:
            return self.active                       # keep current pinch (hysteresis)
        if self.ratios[closest] < PINCH_ON[closest]:
            self.active = closest
        else:
            self.active = None
        return self.active


# ---------- Main ----------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("Could not open the camera. Check camera permissions in System Settings.")
        return

    fx = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
    fy = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)

    pinch_sel = PinchSelector()

    pinch_start = None
    frozen_pos = None
    dragging = False
    last_click = 0
    right_clicked = False
    scroll_anchor = None
    vol_anchor = None
    vol_base = None
    vol_shown = None
    palm_history = deque()
    last_swipe = 0
    swipe_msg_until = 0
    swipe_msg = ""
    mode = "idle"

    fps, fps_t, fps_n = 0.0, time.time(), 0
    fist_start = None
    quit_requested = False

    win = "Gesture Control (q / Esc / hold fist 3s to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    def release_drag():
        nonlocal dragging
        if dragging:
            pyautogui.mouseUp()
            dragging = False

    smoother = LandmarkSmoother(LANDMARK_SMOOTHING)

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2 if PREFERRED_HAND else 1,
                        model_complexity=1,          # the full (most accurate) model
                        min_detection_confidence=DETECTION_CONFIDENCE,
                        min_tracking_confidence=TRACKING_CONFIDENCE) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Camera frame not received. Check camera permissions.")
                break
            frame = cv2.flip(frame, 1)
            if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
                frame = cv2.resize(frame, (CAM_W, CAM_H))
            detect_frame = enhance(frame) if LOW_LIGHT_BOOST else frame
            rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            now = time.time()

            cv2.rectangle(frame, (FRAME_MARGIN, FRAME_MARGIN),
                          (CAM_W - FRAME_MARGIN, CAM_H - FRAME_MARGIN), (255, 0, 255), 2)

            # Pick the hand to follow
            hand, label = None, None
            if result.multi_hand_landmarks:
                for h, hd in zip(result.multi_hand_landmarks, result.multi_handedness):
                    lbl = hd.classification[0].label
                    if PREFERRED_HAND is None or lbl == PREFERRED_HAND:
                        hand, label = h, lbl
                        break

            if hand is not None:
                lm = hand.landmark
                smoother(lm)
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                up = fingers_up(lm, label)
                active_pinch = pinch_sel.update(lm)
                p_idx = active_pinch == "index"
                p_mid = active_pinch == "middle"
                p_ring = active_pinch == "ring"

                if SHOW_DEBUG:
                    names = ("index", "middle", "ring", "pinky")
                    txt = "  ".join(
                        f"{n[0].upper()}:{'UP' if up[n] else '--'} {last_straightness[n]:.2f}"
                        for n in names)
                    cv2.putText(frame, "straight " + txt, (10, CAM_H - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    ptxt = "  ".join(f"{n}:{pinch_sel.ratios[n]:.2f}"
                                     for n in ("index", "middle", "ring"))
                    cv2.putText(frame, f"pinch {ptxt}  -> {active_pinch or 'none'}",
                                (10, CAM_H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 2)

                four_up = all(up[f] for f in ("index", "middle", "ring", "pinky"))
                none_up = not any(up[f] for f in ("index", "middle", "ring", "pinky"))

                # Reset state that belongs to other modes
                if not (up["index"] and up["middle"] and not up["ring"] and not p_idx):
                    scroll_anchor = None
                if not p_ring:
                    vol_anchor = None
                if not p_mid:
                    right_clicked = False
                if not four_up:
                    palm_history.clear()

                # --- Fist: pause ---
                if none_up and not (p_idx or p_mid or p_ring):
                    if fist_start is None:
                        fist_start = now
                    held = now - fist_start
                    if held >= QUIT_FIST_SECONDS:
                        quit_requested = True
                    mode = f"paused (quit in {max(0, QUIT_FIST_SECONDS - held):.1f}s)" \
                        if held > 1.0 else "paused"
                    release_drag()
                    pinch_start = None
                    frozen_pos = None
                    fx.reset(); fy.reset()

                # --- Volume: thumb + ring pinch, move up/down ---
                elif p_ring and not p_idx and not p_mid:
                    mode = "volume"
                    y = lm[0].y
                    if vol_anchor is None:
                        vol_anchor = y
                        vol_base = get_volume()
                        vol_shown = vol_base
                    delta = (vol_anchor - y) * VOLUME_SENSITIVITY
                    if IS_MAC:
                        target = max(0, min(100, vol_base + delta))
                        if abs(target - vol_shown) >= 2:
                            vol_shown = set_volume(target)
                    else:
                        if abs(delta) >= 5:
                            nudge_volume_keys(delta)
                            vol_anchor = y
                    if vol_shown is not None and IS_MAC:
                        bar = int(vol_shown * 2)
                        cv2.rectangle(frame, (10, 60), (210, 80), (80, 80, 80), 1)
                        cv2.rectangle(frame, (10, 60), (10 + bar, 80), (0, 200, 255), -1)
                        cv2.putText(frame, f"{int(vol_shown)}%", (220, 78),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                # --- Right click: thumb + middle ---
                elif p_mid and not p_idx:
                    mode = "right click"
                    if not right_clicked and now - last_click > CLICK_COOLDOWN:
                        pyautogui.rightClick()
                        last_click = now
                        right_clicked = True

                # --- Open palm: swipe detection ---
                elif four_up and not p_idx:
                    mode = "palm (swipe)"
                    palm_x = lm[9].x
                    palm_history.append((now, palm_x))
                    while palm_history and now - palm_history[0][0] > SWIPE_WINDOW:
                        palm_history.popleft()
                    if len(palm_history) >= 3 and now - last_swipe > SWIPE_COOLDOWN:
                        dx = palm_x - palm_history[0][1]
                        if abs(dx) > SWIPE_DISTANCE:
                            # Hand moves left -> next desktop (like a trackpad swipe)
                            direction = 1 if dx < 0 else -1
                            switch_desktop(direction)
                            last_swipe = now
                            palm_history.clear()
                            swipe_msg = "Next desktop" if direction > 0 else "Previous desktop"
                            swipe_msg_until = now + 0.8

                # --- Scroll: index + middle up ---
                elif up["index"] and up["middle"] and not up["ring"] and not p_idx:
                    mode = "scroll"
                    y = lm[8].y
                    if scroll_anchor is None:
                        scroll_anchor = y
                    delta = scroll_anchor - y
                    if abs(delta) > 0.02:
                        pyautogui.scroll(int(delta * SCROLL_SENSITIVITY * 10))
                        scroll_anchor = y

                # --- Move / click / drag ---
                else:
                    sx, sy = map_to_screen(lm[8].x, lm[8].y)
                    cx, cy = fx(sx, now), fy(sy, now)

                    if p_idx:
                        if pinch_start is None:
                            pinch_start = now
                            frozen_pos = pyautogui.position()   # lock cursor for a clean click
                        elif not dragging and now - pinch_start > DRAG_HOLD_TIME:
                            pyautogui.mouseDown()
                            dragging = True
                        if dragging:
                            mode = "drag"
                            pyautogui.moveTo(cx, cy)
                        else:
                            mode = "pinch"
                    else:
                        if pinch_start is not None:
                            if dragging:
                                release_drag()
                            elif now - last_click > CLICK_COOLDOWN:
                                pyautogui.click(*frozen_pos)
                                last_click = now
                            pinch_start = None
                            frozen_pos = None
                        if up["index"]:
                            mode = "move"
                            pyautogui.moveTo(cx, cy)
                        else:
                            mode = "idle"
            else:
                mode = "no hand"
                smoother.reset()
                release_drag()
                pinch_start = None
                frozen_pos = None
                scroll_anchor = None
                vol_anchor = None
                palm_history.clear()
                fx.reset(); fy.reset()

            # FPS
            fps_n += 1
            if now - fps_t >= 0.5:
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0

            cv2.putText(frame, f"Mode: {mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.0f}", (CAM_W - 110, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if now < swipe_msg_until:
                cv2.putText(frame, swipe_msg, (10, CAM_H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

            if not mode.startswith("paused"):
                fist_start = None

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or quit_requested:   # 27 = Esc
                break
            # Closing the window with its red button also quits
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

    release_drag()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)          # lets macOS actually close the window
    print("Gesture control stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:          # Ctrl+C in the terminal
        pyautogui.mouseUp()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("\nGesture control stopped.")