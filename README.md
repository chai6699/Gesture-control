# Gesture Control

Control your computer with hand gestures using your webcam. Built with Python, [MediaPipe](https://developers.google.com/mediapipe) hand tracking, OpenCV and PyAutoGUI.

## Gestures

| Gesture | Action |
|---|---|
| Index finger up | Move cursor |
| Quick pinch, thumb + index | Left click |
| Held pinch, thumb + index | Drag |
| Pinch, thumb + middle | Right click |
| Index + middle up, move up/down | Scroll |
| Pinch thumb + ring, move up/down | Volume (macOS) |
| Open palm, fast swipe left/right | Switch desktop |
| Fist | Pause |
| Fist held for 3 seconds | Quit |

You can also quit with `q`, `Esc`, or `Ctrl+C` in the terminal.

## Features

- One Euro filter for a cursor that is steady when slow and responsive when fast
- Cursor freezes during a pinch so clicks land where you aimed
- Pinch and finger detection that adapt to hand size and angle
- Landmark smoothing and low-light enhancement for stable tracking
- On-screen debug readout for tuning

## Setup

Requires Python 3.9–3.12 (MediaPipe doesn't support newer versions yet).

```bash
git clone https://github.com/<your-username>/gesture-control.git
cd gesture-control
pip install -r requirements.txt
python gesture_control.py
```

Using conda:

```bash
conda create -n gesture python=3.12 -y
conda activate gesture
pip install -r requirements.txt
python gesture_control.py
```

### macOS permissions

Allow your terminal or editor (Terminal, VS Code, etc.) under **System Settings → Privacy & Security**:
- **Camera**, so it can see your hand
- **Accessibility**, so it can move the cursor

## Tuning

All settings are at the top of `gesture_control.py`:

| Setting | What it does |
|---|---|
| `FILTER_BETA` | Raise if the cursor lags, lower if it jitters |
| `LANDMARK_SMOOTHING` | Raise for steadier tracking, lower for less lag |
| `PINCH_ON` | Raise a finger's value if its pinch doesn't register |
| `FINGER_ON` | Lower if raised fingers aren't detected |
| `SWIPE_DISTANCE` | Lower if swipes are hard to trigger |
| `PREFERRED_HAND` | Set to `"Right"` or `"Left"` to track only that hand |
| `SHOW_DEBUG` | Show or hide the debug numbers |

## Safety

PyAutoGUI's failsafe is on: move the real mouse into a screen corner to stop the script immediately.
