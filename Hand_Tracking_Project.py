#TODO: These are the libraries and modules imported into the program. Each one provides different functions that the game needs.
import math
import os
import time
import random

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


# TODO: These are constant variables used for the hand-tracking setup
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

CAM_SOURCE = 0
FRAME_W, FRAME_H = 1280, 720

THUMB_TIP_ID = 4
INDEX_TIP_ID = 8
PINCH_THRESHOLD = 40

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


# TODO:These variables store the values used by the game, such as colors, size, health, and damage.
SKELETON_COLOR, SKELETON_GLOW = (92, 82, 5), (255, 200, 130)  
JOINT_COLOR = (255, 100, 60)   
JOINT_GLOW = (255, 255, 100)   

SLIME_RADIUS = 35
SLIME_COLOR, SLIME_GLOW = (100, 255, 100), (150, 255, 150)
MAX_HEALTH = 100

PARTICLE_COLOR = (120, 255, 150)   
PARTICLE_LIFE = 18                 
PARTICLE_SPAWN_PER_FRAME = 10      


BLOCK_SIZE = 50
BLOCK_COLOR, BLOCK_GLOW = (100, 100, 255), (150, 150, 255)   
BLOCK_SPAWN_RATE = 20
BLOCK_DAMAGE = 5


# TODO: These functions are used to draw glowing lines and circles to make the hand skeleton and joints.
def glow_line(img, p1, p2, color, glow, thickness=2):
    cv2.line(img, p1, p2, glow, thickness + 6, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

def glow_circle(img, center, radius, color, glow):
    cv2.circle(img, center, radius + 8, glow, -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)


#TODO: This function draws text and a background rectangle on the screen.
def text_with_bg(img, text, org, scale=0.9, color=(255, 255, 255), bg=(0, 0, 0), alpha=0.55):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 2)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 10, y - th - 10), (x + tw + 10, y + baseline + 6), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, scale, color, 2, cv2.LINE_AA)


#TODO: Check if MediaPipe hand-tracking model file is downloaded
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand landmark model (one-time, ~8 MB)...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


#TODO: Particles blueprint that is created when the slime moves. It has a position, direction, speed, lifetime, and size.
class Particle:
    #TODO: This part sets up each particle by giving it a position, random direction, speed, lifetime, and size.
    def __init__(self, x, y):
        self.x, self.y = x, y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, 2.5)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = PARTICLE_LIFE
        self.max_life = PARTICLE_LIFE
        self.radius = random.randint(3, 6)

    #TODO: It moves the particle and decreases its lifetime by 1 frame until it disappears.
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.15  
        self.life -= 1
        return self.life > 0

    #TODO: This method displays the particle and makes it fade and shrink before it disappears.
    def draw(self, img):
        fade = self.life / self.max_life
        radius = max(1, int(self.radius * fade))
        color = tuple(int(c * fade) for c in PARTICLE_COLOR)
        cv2.circle(img, (int(self.x), int(self.y)), radius, color, -1, cv2.LINE_AA)



#TODO: Slime blueprint that moves toward the finger, has health, and creates particles when it moves. It can also be hit by blocks.
class Slime:
    #TODO: This sets up the slime with its position, health, and particles.
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.health = MAX_HEALTH
        self.particles = []

    #TODO: It moves the slime toward the finger when the distance is more than 2, creates particles while the slime moves, and updates the particles.
    def update(self, target_x, target_y):
        dx, dy = target_x - self.x, target_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 2:
            speed = min(dist * 0.15, 25)
            self.x += dx / dist * speed
            self.y += dy / dist * speed

            for _ in range(PARTICLE_SPAWN_PER_FRAME):
                self.particles.append(Particle(self.x, self.y))
        self.age_particles()

    #TODO: It removes particles that have reached the end of their lifetime.
    def age_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    #TODO: It draws the particles, the moving/glowing slime, and its eyes.
    def draw(self, img):
        for p in self.particles:
            p.draw(img)

        jiggle = int(3 * math.sin(time.time() * 8))
        radius = SLIME_RADIUS + jiggle
        center = (int(self.x), int(self.y))
        glow_circle(img, center, radius, SLIME_COLOR, SLIME_GLOW)
        for dx in (-15, 15):
            cv2.circle(img, (center[0] + dx, center[1] - 15), 4, (0, 0, 0), -1)  

    #TODO: Damage of the box in slime
    def hit_by(self, block):
        return math.hypot(self.x - block.x, self.y - block.y) < (SLIME_RADIUS + BLOCK_SIZE)



#TODO:  Block blueprint that falls from the top of the screen, has a position and speed, and can hit the slime.
class Block:
    #TODO: Creates/sets up for the Block. position and the speed.
    def __init__(self, x, speed):
        self.x, self.y, self.speed = x, -BLOCK_SIZE, speed

    #TODO: It moves the block down and checks if it is already out of the screen.
    def update(self, screen_h):
        self.y += self.speed
        return self.y > screen_h

    #TODO: Draw the block on the screen with a glow effect.
    def draw(self, img):
        half = BLOCK_SIZE // 2
        x1, y1, x2, y2 = self.x - half, self.y - half, self.x + half, self.y + half
        cv2.rectangle(img, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), BLOCK_GLOW, 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), BLOCK_COLOR, -1)


#TODO: It detects the hand, tracks the fingers, checks for pinching, stores the hand information, and returns it so other functions can use it in the game.
def read_hands(result, w, h, overlay):
    hands = []
    if not result.hand_landmarks:
        return hands

    for i, landmarks in enumerate(result.hand_landmarks[:2]):
        pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]

        for a, b in HAND_CONNECTIONS:
            glow_line(overlay, pts[a], pts[b], SKELETON_COLOR, SKELETON_GLOW, 2)
        for p in pts:
            glow_circle(overlay, p, 4, JOINT_COLOR, JOINT_GLOW)

        pinch_dist = math.hypot(pts[THUMB_TIP_ID][0] - pts[INDEX_TIP_ID][0],
                                 pts[THUMB_TIP_ID][1] - pts[INDEX_TIP_ID][1])
        is_pinching = pinch_dist < PINCH_THRESHOLD
        label = result.handedness[i][0].category_name

        hands.append({"label": label, "index_tip": pts[INDEX_TIP_ID], "is_pinching": is_pinching})

        status = f"{label}: {'PINCHING' if is_pinching else 'OPEN'}"
        color = (0, 255, 0) if is_pinching else (0, 100, 255) 
        cv2.putText(overlay, status, (20, 80 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return hands

#TODO: To choose which hand will control the slime.
def pick_control_hand(hands):
    right_hand = None
    left_hand = None

    for hand in hands:
        if hand["label"] == "Right":
            right_hand = hand
        elif hand["label"] == "Left":
            left_hand = hand

    is_right_pinching = right_hand is not None and right_hand["is_pinching"]
    is_left_pinching = left_hand is not None and left_hand["is_pinching"]

    if is_right_pinching:
        return right_hand

    if is_left_pinching:
        return left_hand

    if right_hand is not None:
        return right_hand

    if left_hand is not None:
        return left_hand

    return None


#TODO: Main
def main():
    ensure_model()

    # TODO: It sets up the MediaPipe hand detector with the needed settings.
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )
    # TODO: This part opens the webcam, sets the camera resolution, checks if the webcam is working, and displays the game in fullscreen.
    cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_DSHOW) if isinstance(CAM_SOURCE, int) else cv2.VideoCapture(CAM_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAM_SOURCE / camera permissions")

    cv2.namedWindow("Hand Slime Dodger", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Hand Slime Dodger", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
   

    #TODO: This part sets up the game variables, such as the game state, countdown, slime, blocks, timers, and game time.
    state = "WAITING"
    countdown_start = None
    slime = None
    blocks = []
    spawn_timer = 0
    game_start_time = None
    elapsed = 0
    prev_time = time.time()

    #TODO: Control pressed q or Q to quit the game.
    print("Hand Slime Dodger running. Press 'q' or 'Q' to quit.")


    #TODO: Keep taking pictures from the webcam. Stop if the camera cannot give a picture.
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        overlay = np.zeros_like(frame)
        hands = read_hands(result, w, h, overlay)
        control_hand = pick_control_hand(hands)


        # TODO: This part controls the game mechanics. It manages the different game states: waiting, ready, 
        #  countdown, playing, and game over. It uses the hand pinch to start and control the slime, creates 
        #  and moves the blocks, checks if the blocks hit the slime and reduces its health, shows the game 
        #  information, and allows the player to restart or quit.

        if state == "WAITING":
            if hands:
                state = "READY"
            text_with_bg(overlay, "SHOW YOUR HANDS", (w // 2 - 200, 100), 1.2, (0, 100, 255)) 

        elif state == "READY":
            if not hands:
                state = "WAITING"
            else:
                text_with_bg(overlay, "PINCH YOUR HAND TO START", (w // 2 - 200, 100), 1.2, (0, 255, 255)) 
                if control_hand and control_hand["is_pinching"]:
                    state = "COUNTDOWN"
                    countdown_start = time.time()

        elif state == "COUNTDOWN":
            if not hands:
                state = "WAITING"
            else:
                remaining = 3 - int(time.time() - countdown_start)
                if remaining <= 0:
                    state = "PLAYING"
                    slime = Slime(*(control_hand or hands[0])["index_tip"])
                    blocks, spawn_timer = [], 0
                    game_start_time, elapsed = time.time(), 0
                else:
                    text_with_bg(overlay, str(remaining), (w // 2 - 40, h // 2), 3.0,
                                 (0, 255, 255), (50, 50, 100))  

        elif state == "PLAYING":
            elapsed = time.time() - game_start_time
            text_with_bg(overlay, f"Time: {int(elapsed)}s", (w // 2 - 60, 100), 0.9, (255, 255, 100))  

            spawn_timer += 1
            if spawn_timer >= BLOCK_SPAWN_RATE:
                spawn_timer = 0
                blocks.append(Block(random.randint(BLOCK_SIZE, w - BLOCK_SIZE), random.randint(4, 7)))

            blocks = [b for b in blocks if not b.update(h)]
            for b in blocks:
                b.draw(overlay)

            if control_hand and control_hand["is_pinching"]:
                slime.update(*control_hand["index_tip"])
            else:
                slime.age_particles()

            for b in blocks:
                if slime.hit_by(b):
                    slime.health -= BLOCK_DAMAGE
            slime.draw(overlay)

            status = "CONTROLLING" if control_hand and control_hand["is_pinching"] else ("LOST" if not control_hand else "RELEASED")
            label_x, label_y = w // 2 - 100, h - 75
            cv2.putText(overlay, f"Player {status}", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SLIME_COLOR, 2)
            cv2.putText(overlay, f"HP: {max(0, int(slime.health))}/100", (label_x, label_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, SLIME_COLOR, 2)

            if slime.health <= 0:
                state = "GAME_OVER"

        elif state == "GAME_OVER":
            text_with_bg(overlay, "GAME OVER!", (w // 2 - 150, h // 2 - 150), 2.0,
                         (0, 0, 255), (100, 0, 0))  
            text_with_bg(overlay, f"Survived: {int(elapsed)} seconds", (w // 2 - 180, h // 2 - 50), 1.2,
                         (255, 200, 0), (50, 50, 0))  
            text_with_bg(overlay, "Press R or r to restart", (w // 2 - 160, h // 2 + 50), 0.9, (255, 255, 255)) 
            text_with_bg(overlay, "Press Q or q to quit", (w // 2 - 160, h // 2 + 80), 0.9, (255, 255, 255)) 


        frame = cv2.addWeighted(frame, 0.7, overlay, 1.0, 0)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        text_with_bg(frame, "Hand Slime Dodger", (20, 40), 1.0, (0, 255, 255))  
        text_with_bg(frame, f"FPS: {fps:0.0f}", (20, h - 20), 0.6, (200, 200, 200), alpha=0.4)  

        cv2.imshow("Hand Slime Dodger", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("r"), ord("R")) and state == "GAME_OVER":
            state, slime, game_start_time, elapsed = "WAITING", None, None, 0

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()