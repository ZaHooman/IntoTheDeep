# 🌍 Into The Deep
### A Hand-Gesture Controlled Adventure Game
**🏆 1st Place — Hack Club Campfire Hackathon, Chester County**

---

## What is it?

Beneath the Surface is a 2D adventure game controlled entirely by your hands through a webcam. You don't need a controller, nor a keyboard, just your hands!

The player guides Lani through a world that starts above ground and descends deeper and deeper into the Earth — through the hurdled surfaces, fossil layers, and the unknown below. Each level goes further down, with new environments and mechanics.

All graphics are hand-drawn by our incredibly graphics artist!

---

## Why we built it

Gaming should be for everyone.

- 🎮 **Controllers cost $70+** — a webcam is already built into most laptops
- ⌨️ **Keyboards aren't accessible for everyone** — hand gestures can be customized to fit whatever movements a player is capable of

---

## Levels

| Level | Location | Vibe |
|-------|----------|------|
| 1 | The Surface | Jump over obstacles, explore the land of the sun before it's gone |
| 2 | The Mines | Fall into the earth, darker and tighter, new puzzle mechanics |
| ... | Going deeper | Each level descends further underground |

---

## Controls

| Gesture | Action |
|---------|--------|
| ☝️ Point finger | Move in that direction (up / down / left / right) |
| ✋ Open hand | Cursor mode — aim at UI elements |
| ✊ Fist | Grab and drag objects |
| ✋ Hover (multple seconds) | Activate buttons — hold your cursor over anything, wait for the progress bar to finish |

---

## Tech Stack

- **Python 3.10** — Python version that works best with other softwares
- **MediaPipe 0.10.13** — hand landmark detection via the Tasks API
- **OpenCV** — webcam capture and frame processing
- **Pygame** — game engine and rendering
- **NumPy** — pixel conversion between OpenCV and Pygame
- **Threading** — camera and game loop run in parallel
- **Procreate** — art app to develop custom sprites

---

## How to Run

**1. Install dependencies**
```bash
pip install mediapipe==0.10.13 pygame opencv-python numpy
```

**2. Download the hand tracking model**
```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
Place `hand_landmarker.task` in the same folder as the game.

**3. Run**
```bash
python Main.py
```

Press **ESC** or the **X** button to quit.

---

## File Structure

```
/
├── Main.py                 # Main game file
├── hand_landmarker.task    # MediaPipe model (download separately)
├── Images                  # Background image
└── Assets                  # Font files & music files
```

---

## How it works

The game runs two threads simultaneously:

- **Camera thread** — reads webcam frames, runs MediaPipe hand detection, updates shared gesture/direction/cursor state
- **Game thread** — runs the Pygame loop at 60fps, reads shared state to move the player and handle interactions

The hand skeleton is rendered as a glowing overlay in the top-right corner of the game window — no separate webcam window needed.

---

## Developers

Our amazing team!

- **Aishi Garimella** — Graphics Designer
- **Raksha Kumaresan** - ML and Python Game Developer
- **Sravani Mahankali** - ML and Python Game Developer

---

## Built at

**Hack Club Campfire Hackathon — Chester County**
🏆 1st Place Winner
