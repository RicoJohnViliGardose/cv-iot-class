import ctypes
import math
import os
import random
import sys
import time

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print(f"ERROR: Missing dependency ({exc.name}).")
    print("Install everything with:  pip install opencv-python mediapipe numpy")
    sys.exit(1)

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    print("ERROR: MediaPipe is not installed.")
    print("Install it with:  pip install mediapipe")
    sys.exit(1)
except AttributeError:
    print("ERROR: This MediaPipe version does not provide 'mp.solutions.face_mesh'.")
    print("Try:  pip install \"mediapipe<0.10.20\"")
    sys.exit(1)



#TODO:CONFIG

WINDOW_NAME = "FACE DIMENSION GAME"

CAM_SOURCE = 0
#TODO: Laptop with a build- in webcam: leave as the drive index above (0).
#TODO: Lab PC with no webcam: set to your phone's IP Webcam stream tunneled over
#TODO: USB via 'asb forward tcp:8080 tcp:8080 ', e.g.:
#TODO: CAM_SOURCE = "http://localhost:8080/video"

#TODO: WHERE TO RECORD SCREENSHOTS
SCREENSHOT_DIR = "screenshots"
FRAME_W, FRAME_H = 1280, 720
FONT = cv2.FONT_HERSHEY_SIMPLEX

#TODO: THE COLORS USED THROUGHOUT THE GAME
THEME = {"primary": (255, 255, 0), "accent": (0, 255, 255), "mesh": (255, 200, 0)}
DIMENSION_TITLES = [
    "DIMENSION 1: FILTERS + FALLING WORDS",
    "DIMENSION 2: MISSING LETTERS",
]
GOOD_COLOR = (0, 230, 0)
BAD_COLOR = (60, 60, 255)
GOLD_COLOR = (0, 215, 255)
WHITE = (255, 255, 255)
FALL_COLORS = [(255, 255, 0), (255, 0, 255), (0, 255, 150), (0, 200, 255), (255, 150, 0)]

BRACKET_LEN = 30
BRACKET_THICKNESS = 3

#TODO: FACIAL LANDMARK THAT DETECTS EACH FEATURE (from MediaPipe Face Mesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]    
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380] 
LM_MOUTH_LEFT = 61
LM_MOUTH_RIGHT = 291
LM_MOUTH_TOP = 13
LM_MOUTH_BOTTOM = 14
LM_NOSE_TIP = 1
LM_CHEEK_LEFT = 234
LM_CHEEK_RIGHT = 454
LM_FOREHEAD = 10


# TODO: These settings control how the game detects eye closures, winks, mouth movements, smiles,
# TODO: and head turns. They also set the required timing and cooldowns to prevent accidental actions,
# TODO: such as switching dimensions too quickly or taking multiple photos from one smile.
EAR_CLOSED = 0.19          
EAR_OPEN = 0.23         
CLOSURES_NEEDED = 6        
CLOSURE_MIN_SECONDS = 0.25 
CLOSURE_WINDOW = 15.0    
WINK_FRAMES = 3            
DOUBLE_BLINK_WINDOW = 1.5 
MAR_ON = 0.35           
MAR_OFF = 0.22            
SMILE_ON = 0.45          
SMILE_HOLD = 0.5         
SMILE_PHOTO_COOLDOWN = 3.0
HEAD_TURN = 0.11          
HEAD_RETURN = 0.06       
HEAD_COOLDOWN = 0.6       
LOCK_AFTER_SWITCH = 2.0  

#TODO: DIMENSION 2 Words EATING DISPLAY
DIM2_WORDS = ["fresh", "day", "happy", "world", "dream", "smile", "bright", "hope", "magic", "peace",
              "sunny", "kind", "brave", "calm", "cool", "dance", "joy", "love", "star", "wonder",
              "light", "music", "play", "shine", "sweet", "team", "wild", "young", "cheer", "grow"]
MAX_COLLECTED_WORDS = 12
COLLECT_COOLDOWN = 0.4
LETTER_RADIUS = 20                     
MAX_LETTERS_ON_SCREEN = 24            

MAX_LIVES = 3                          
MAX_WORD_LEN = 14                      

#TODO:It controls the game difficulty by increasing the word length and the number of missing letters as the level goes up
def level_spec(level):
    n = level + 1
    if n <= 10:
        length = n + 2                                
        missing = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3][n - 1]
    else:
        length = min(MAX_WORD_LEN, 12 + (n - 10) // 2)
        missing = min(length - 3, 8, 3 + (n - 10) // 2)
    return length, missing


#TODO:This code stores words and groups them by length so each game level can choose the right words.
_ALL_WORDS = """
sun cat dog sky map pen cup hat bus car fox owl bee ice sea key box toy jam leg
moon star tree fish bird cake rain snow wind door book ship lamp frog milk gold rose lake hill kite
cloud happy dream smile plant river tiger pizza candy music bread chair apple beach horse light stone
planet garden castle window flower orange purple guitar rocket turtle bridge forest silver yellow summer
winter spring friend monkey pencil
rainbow harmony chicken kitchen balloon dolphin mystery journey library penguin pumpkin kingdom diamond
freedom station crystal
mountain notebook question sunshine vacation elephant computer dinosaur umbrella airplane football painting
universe treasure birthday triangle sandwich
adventure beautiful celebrate discovery education fantastic butterfly chocolate waterfall wonderful xylophone
yesterday telescope landscape happiness knowledge
friendship playground strawberry basketball electronic grandmother adventures programmer experience
imagination temperature celebration photography development environment helicopter 
kindergarten conversation intelligence civilization photographer thunderstorm championship transparency
constellation extraordinary international communication determination
responsibility transportation entertainment accomplishment representative overwhelmingly
understanding relationships unforgettable opportunities mathematician uncomfortable
waterproof microscope dictionary everything toothbrush scoreboard revolution volleyball
mathematics butterflies restaurants wheelbarrow grasshopper refrigerator thankfulness strawberries neighborhood
""".split()
WORDS_BY_LEN = {}
for _w in dict.fromkeys(_ALL_WORDS):
    WORDS_BY_LEN.setdefault(len(_w), []).append(_w)


#TODO:This code finds words with the required number of letters. If none are found, it tries shorter words.
def words_for_length(length):
    while length not in WORDS_BY_LEN and length > 3:
        length -= 1
    return WORDS_BY_LEN[length]


#TODO:These are the messages that appear when the player collects a word or letter correctly.
GOOD_MESSAGES = ["Excellent!", "Correct!", "Amazing!", "Great Job!", "Perfect!"]
WORD_PRAISE = ["Yum!", "Nice catch!", "Sweet!", "Awesome!"]

GLASS_W, GLASS_H = 600, 240         
GL_L, GL_R = (170, 120), (430, 120)  
GL_EYE_DIST = GL_R[0] - GL_L[0]      

#TODO: Creates a class for drawing the sunglasses filter.
class Art:
    #TODO:Creates an empty transparent canvas.
    def __init__(self):
        self.P = np.zeros((GLASS_H, GLASS_W, 3), np.float32)  
        self.A = np.zeros((GLASS_H, GLASS_W, 1), np.float32)  
    #TODO:Draws colored shapes on the canvas
    def paint(self, draw_fn, color, opacity=1.0):
        mask = np.zeros((GLASS_H, GLASS_W), np.uint8)
        draw_fn(mask)                                         
        a = mask.astype(np.float32)[..., None] / 255.0 * opacity
        col = color if isinstance(color, np.ndarray) else np.array(color, np.float32)
        self.P = self.P * (1 - a) + col * a
        self.A = self.A * (1 - a) + a
    #TODO:Combines the colors and transparency to create the final sunglasses image
    def finish(self):
        return np.concatenate([self.P, self.A], axis=2)

#TODO: Makes the sunglasses lenses change color from top to bottom.
def vgrad(top, bottom):
    t = np.linspace(0, 1, GLASS_H, dtype=np.float32).reshape(GLASS_H, 1, 1)
    col = np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t
    return col * np.ones((1, GLASS_W, 1), np.float32)

#TODO:Draws a rectangle with rounded corners, either filled with color or showing only its outline.
def draw_round_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)

#TODO:Draws the same shape on both the left and right sunglasses lenses to avoid repeating code.
def _both(fn):
    def draw(m):
        for cx, cy in (GL_L, GL_R):
            fn(m, cx, cy)
    return draw

#TODO: Draws the nose bridge and the two arms of the sunglasses that extend toward the ears.
def _bridge_and_arms(art, color, thick, y=95, x_in=110):
    def draw(m):
        cv2.line(m, (GL_L[0] + x_in - 10, y), (GL_R[0] - x_in + 10, y), 255, thick, cv2.LINE_AA)
        cv2.line(m, (GL_L[0] - x_in, y), (0, y - 8), 255, thick, cv2.LINE_AA)
        cv2.line(m, (GL_R[0] + x_in, y), (GLASS_W, y - 8), 255, thick, cv2.LINE_AA)
    art.paint(draw, color)

#TODO: Creates black sunglasses with dark lenses, shiny details, a nose bridge, and side arms.
def build_black():
    art = Art()
    shape = lambda m, cx, cy, t=-1: draw_round_rect(m, cx - 110, cy - 75, cx + 110, cy + 75, 36, 255, t)
    art.paint(_both(shape), vgrad((20, 20, 20), (70, 60, 60)), 0.94)
    art.paint(_both(lambda m, cx, cy: cv2.line(m, (cx - 65, cy + 45), (cx - 15, cy - 35), 255, 9, cv2.LINE_AA)),
              (255, 255, 255), 0.22)                                   
    art.paint(_both(lambda m, cx, cy: (shape(m, cx, cy, 13),
                                       cv2.line(m, (cx - 112, cy - 72), (cx + 112, cy - 72), 255, 20, cv2.LINE_AA))),
              (12, 12, 12))
    _bridge_and_arms(art, (12, 12, 12), 14)
    return art.finish()

#TODO: Creates aviator sunglasses with colored lenses, gold frames, a nose bridge, and side arms.
def build_aviator():
    art = Art()
    gold = (40, 190, 240)
    art.paint(_both(lambda m, cx, cy: cv2.ellipse(m, (cx, cy + 8), (108, 80), 0, 0, 360, 255, -1, cv2.LINE_AA)),
              vgrad((30, 40, 50), (40, 120, 200)), 0.85)
    art.paint(_both(lambda m, cx, cy: cv2.ellipse(m, (cx, cy + 8), (108, 80), 0, 0, 360, 255, 9, cv2.LINE_AA)), gold)
    _bridge_and_arms(art, gold, 7, y=90, x_in=108)
    return art.finish()

#TODO: Creates colorful round sunglasses with purple and yellow-orange lenses, white frames, a nose bridge, and side arms.
def build_colorpop():
    art = Art()
    art.paint(lambda m: cv2.circle(m, GL_L, 88, 255, -1, cv2.LINE_AA), vgrad((190, 90, 255), (80, 60, 255)), 0.82)
    art.paint(lambda m: cv2.circle(m, GL_R, 88, 255, -1, cv2.LINE_AA), vgrad((255, 230, 80), (255, 120, 30)), 0.82)
    art.paint(_both(lambda m, cx, cy: cv2.circle(m, (cx, cy), 88, 255, 10, cv2.LINE_AA)), (255, 255, 255))
    _bridge_and_arms(art, (255, 255, 255), 9, x_in=88)
    return art.finish()

#TODO: Creates futuristic visor sunglasses with dark lenses, cyan glowing lines, and side arms
def build_visor():
    art = Art()
    cyan = (255, 255, 0)
    visor = lambda m, t=-1: draw_round_rect(m, 40, 70, 560, 172, 48, 255, t)
    art.paint(visor, vgrad((70, 25, 10), (110, 45, 15)), 0.9)
    art.paint(lambda m: cv2.rectangle(m, (70, 108), (530, 124), 255, -1), cyan, 0.5)       
    art.paint(lambda m: visor(m, 18), cyan, 0.28)                                          
    art.paint(lambda m: visor(m, 5), cyan)                                                
    art.paint(lambda m: (cv2.line(m, (40, 110), (0, 100), 255, 7, cv2.LINE_AA),
                         cv2.line(m, (560, 110), (600, 100), 255, 7, cv2.LINE_AA)), cyan)
    return art.finish()

#TODO: Creates heart-shaped sunglasses with purple lenses, shiny highlights, and dark frames.
def build_hearts():
    art = Art()
    def heart(cx, cy, s=6.0):
        t = np.linspace(0, 2 * math.pi, 160)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        return np.stack([cx + x * s, cy - (y + 2.5) * s], 1).astype(np.int32)
    art.paint(_both(lambda m, cx, cy: cv2.fillPoly(m, [heart(cx, cy)], 255, cv2.LINE_AA)),
              vgrad((90, 70, 255), (30, 20, 190)), 0.88)
    art.paint(_both(lambda m, cx, cy: cv2.ellipse(m, (cx - 45, cy - 40), (20, 9), -35, 0, 360, 255, -1, cv2.LINE_AA)),
              (255, 255, 255), 0.45)
    art.paint(_both(lambda m, cx, cy: cv2.polylines(m, [heart(cx, cy)], True, 255, 10, cv2.LINE_AA)), (20, 10, 120))
    art.paint(lambda m: (cv2.line(m, (262, 92), (338, 92), 255, 10, cv2.LINE_AA),
                         cv2.line(m, (78, 92), (0, 84), 255, 8, cv2.LINE_AA),
                         cv2.line(m, (522, 92), (600, 84), 255, 8, cv2.LINE_AA)), (20, 10, 120))
    return art.finish()

#TODO: Creates funny googly-eye sunglasses with white lenses, black pupils, and dark frames.
def build_googly():
    art = Art()
    black = (15, 15, 15)
    art.paint(_both(lambda m, cx, cy: cv2.circle(m, (cx, cy), 92, 255, -1, cv2.LINE_AA)), (250, 250, 250))
    art.paint(_both(lambda m, cx, cy: cv2.circle(m, (cx + 20, cy + 30), 36, 255, -1, cv2.LINE_AA)), black)
    art.paint(_both(lambda m, cx, cy: cv2.circle(m, (cx + 10, cy + 18), 10, 255, -1, cv2.LINE_AA)), (255, 255, 255))
    art.paint(_both(lambda m, cx, cy: cv2.circle(m, (cx, cy), 92, 255, 12, cv2.LINE_AA)), black)
    _bridge_and_arms(art, black, 14, x_in=92)
    return art.finish()


#TODO:This code lists all sunglasses filters, counts how many filters are available, and stores created filters so they can be reused.
GLASS_STYLES = [
    ("BLACK SHADES", build_black),
    ("AVIATOR GOLD", build_aviator),
    ("COLOR POP", build_colorpop),
    ("CYBER VISOR", build_visor),
    ("FUNNY HEARTS", build_hearts),
    ("GOOGLY EYES", build_googly),
    ("NO FILTER", None),
]
NUM_FILTERS = len(GLASS_STYLES)
_GLASS_CACHE = {}

#TODO: This code gets the selected sunglasses filter, creates it if it is not saved yet, and reuses the saved filter to save time.
def get_glasses(idx):
    builder = GLASS_STYLES[idx][1]
    if builder is None:
        return None
    if idx not in _GLASS_CACHE:
        _GLASS_CACHE[idx] = builder()
    return _GLASS_CACHE[idx]

#TODO: Places the sunglasses on the face, adjusting their size, position, and angle to match the eyes.
def draw_glasses(frame, eye_l, eye_r, art):
    h, w = frame.shape[:2]
    eye_l = np.asarray(eye_l, np.float32)
    eye_r = np.asarray(eye_r, np.float32)
    d = eye_r - eye_l
    dist = float(np.linalg.norm(d))
    if dist < 8:
        return
    s = dist / GL_EYE_DIST * 1.1                         
    ang = math.atan2(float(d[1]), float(d[0]))          
    c, sn = s * math.cos(ang), s * math.sin(ang)
    M = np.array([[c, -sn, 0.0], [sn, c, 0.0]], np.float32)
    mid_dst = (eye_l + eye_r) / 2.0                   
    mid_src = (np.array(GL_L, np.float32) + np.array(GL_R, np.float32)) / 2.0
    M[:, 2] = mid_dst - M[:, :2] @ mid_src

    corners = np.array([[0, 0], [GLASS_W, 0], [GLASS_W, GLASS_H], [0, GLASS_H]], np.float32)
    tc = corners @ M[:, :2].T + M[:, 2]
    x0, y0 = max(0, int(math.floor(tc[:, 0].min()))), max(0, int(math.floor(tc[:, 1].min())))
    x1, y1 = min(w, int(math.ceil(tc[:, 0].max()))), min(h, int(math.ceil(tc[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return
    M[:, 2] -= (x0, y0)                                 
    out = cv2.warpAffine(art, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = out[..., 3:4]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(roi * (1.0 - alpha) + out[..., :3], 0, 255).astype(np.uint8)

#TODO: Smooths eye positions to make the sunglasses move more steadily with the face.
def smooth_eyes(app, slot, info):
    cur = np.array([info.eye_l, info.eye_r], np.float32)
    old = app.eye_smooth.get(slot)
    new = cur if old is None else old * 0.4 + cur * 0.6
    app.eye_smooth[slot] = new
    return new[0], new[1]


#TODO: DRAW HELPERS

#TODO: Draws a glowing line with a white highlight to make it look brighter.
def draw_glow_line(img, pt1, pt2, color, thickness=1):
    cv2.line(img, pt1, pt2, color, thickness + 3, cv2.LINE_AA)
    cv2.line(img, pt1, pt2, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

#TODO: Draws glowing corner lines around the face to create a scanning effect.
def draw_corner_brackets(img, x1, y1, x2, y2, color):
    corners = [
        ((x1, y1), (1, 0), (0, 1)),
        ((x2, y1), (-1, 0), (0, 1)),
        ((x1, y2), (1, 0), (0, -1)),
        ((x2, y2), (-1, 0), (0, -1)),
    ]
    for (cx, cy), dx, dy in corners:
        p1 = (cx + dx[0] * BRACKET_LEN, cy)
        p2 = (cx, cy)
        p3 = (cx, cy + dy[1] * BRACKET_LEN)
        draw_glow_line(img, p1, p2, color, BRACKET_THICKNESS)
        draw_glow_line(img, p2, p3, color, BRACKET_THICKNESS)

#TODO: Displays text on the screen with a black outline to make it easier to read.
def draw_hud_text(img, text, org, color, scale=0.55, thickness=1):
    org = (int(org[0]), int(org[1]))
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)

#TODO: Draws a moving horizontal line on the screen to create a scanning effect.
def draw_scan_line(img, frame_count, color, top, bottom, left, right):
    span = bottom - top
    if span <= 0:
        return
    offset = frame_count % 60
    y = top + int((offset / 60) * span)
    cv2.line(img, (left, y), (right, y), color, 1, cv2.LINE_AA)

#TODO: Measures the width and height of text so it can be positioned correctly on the screen.
def text_size(text, scale, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return tw, th

#TODO: Displays text in the center of the screen.
def draw_text_center(img, text, cx, cy, color, scale=0.6, thickness=1):
    tw, th = text_size(text, scale, thickness)
    draw_hud_text(img, text, (cx - tw / 2, cy + th / 2), color, scale, thickness)

#TODO: Reduces the text size until it fits within the available width.
def fit_scale(text, scale, thickness, max_w):
    while scale > 0.3 and text_size(text, scale, thickness)[0] > max_w:
        scale -= 0.05
    return scale

#TODO: Draws a transparent dark panel with a colored border on the screen.
def draw_panel(img, x1, y1, x2, y2, border, alpha=0.6):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    img[y1:y2, x1:x2] = cv2.convertScaleAbs(img[y1:y2, x1:x2], alpha=1.0 - alpha)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 2, cv2.LINE_AA)

#TODO: Draws elements on the screen with adjustable transparency.
def blend_draw(img, alpha, draw_fn):
    if alpha <= 0:
        return
    layer = img.copy()
    draw_fn(layer)
    cv2.addWeighted(layer, min(1.0, alpha), img, 1.0 - min(1.0, alpha), 0, dst=img)

#TODO: Brings the game window to the front so the user can control it using the keyboard.
def force_window_focus(window_title):
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if hwnd:
            user32.ShowWindow(hwnd, 5)
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass 

#TODO: Saves an image as a PNG file in the screenshots folder and avoids overwriting existing files.
def save_image(img, prefix):
    try:
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)     
    except OSError:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"{prefix}_{stamp}.png")
    n = 1
    while os.path.exists(path):      
        path = os.path.join(SCREENSHOT_DIR, f"{prefix}_{stamp}_{n:02d}.png")
        n += 1
    return path if cv2.imwrite(path, img) else None



#TODO:  FACE ANALYSIS (landmarks -> numbers)

#TODO: Creates an empty container to store the face measurements calculated by analyze_face().
class FaceInfo:
    pass

#TODO: Calculates how open or closed the eyes are using facial landmarks.
def eye_aspect_ratio(pt, idxs):
    p = [pt(i) for i in idxs]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return vertical / (2.0 * horizontal + 1e-6)

#TODO: Analyzes facial landmarks to measure eye positions, head direction, mouth opening, smile, and face size, then stores the results in a FaceInfo object.
def analyze_face(landmarks, img_w, img_h):

    def pt(idx):
        lm = landmarks.landmark[idx]
        return np.array([lm.x * img_w, lm.y * img_h], dtype=np.float32)

    info = FaceInfo()
    info.landmarks = landmarks
    info.ear_l = float(eye_aspect_ratio(pt, LEFT_EYE_IDX))
    info.ear_r = float(eye_aspect_ratio(pt, RIGHT_EYE_IDX))
    info.eye_l = tuple(np.mean([pt(i) for i in LEFT_EYE_IDX], axis=0))     
    info.eye_r = tuple(np.mean([pt(i) for i in RIGHT_EYE_IDX], axis=0))

    cheek_l, cheek_r = pt(LM_CHEEK_LEFT), pt(LM_CHEEK_RIGHT)
    face_w = float(np.linalg.norm(cheek_r - cheek_l)) + 1e-6
    nose = pt(LM_NOSE_TIP)

    info.yaw = float((nose[0] - (cheek_l[0] + cheek_r[0]) / 2.0) / face_w)

    mouth_l, mouth_r = pt(LM_MOUTH_LEFT), pt(LM_MOUTH_RIGHT)
    top, bottom = pt(LM_MOUTH_TOP), pt(LM_MOUTH_BOTTOM)
    mouth_w = float(np.linalg.norm(mouth_r - mouth_l)) + 1e-6
    info.mar = float(np.linalg.norm(bottom - top) / mouth_w)     
    info.smile = mouth_w / face_w                              
    center = (top + bottom) / 2.0
    info.mouth = (int(center[0]), int(center[1]))

    pts = np.array([[lm.x * img_w, lm.y * img_h] for lm in landmarks.landmark])
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    info.bbox = (max(0, int(x1) - 15), max(0, int(y1) - 15),
                 min(img_w - 1, int(x2) + 15), min(img_h - 1, int(y2) + 15))
    info.face_w = face_w
    info.cx = float((x1 + x2) / 2)
    info.area = float((x2 - x1) * (y2 - y1))
    forehead = pt(LM_FOREHEAD)
    info.head_top = (float(forehead[0]), float(y1))
    return info

#TODO: Tracks facial gestures, such as blinking, winking, opening the mouth, smiling, and turning the head.
class GestureTracker:
    #TODO: Initializes the gesture-tracking variables.
    def __init__(self):
        self.smile_t0 = None
        self.smile_latched = False
        self.reset()
    #TODO: Resets gesture-tracking states and counters.
    def reset(self, keep_smile=False):
        smile = (self.smile_t0, self.smile_latched) if keep_smile else (None, False)
        self.l_closed = False
        self.r_closed = False
        self.both_t0 = None
        self.both_counted = False        
        self.closures = []            
        self.l_frames = 0
        self.r_frames = 0
        self.l_latched = False
        self.r_latched = False
        self.blinks_l = []              
        self.mouth_open = False
        self.smile_t0, self.smile_latched = smile
        self.yaw = 0.0
        self.head_state = 0
        self.head_armed = False     
        self.head_cd = 0.0

    #TODO: Checks whether both eyes are closed.
    @property
    def both_closed(self):
        return self.l_closed and self.r_closed
    
    #TODO: Counts how many times both eyes were closed within a time limit.
    def closure_count(self, now):
        return len([t for t in self.closures if now - t <= CLOSURE_WINDOW])
    
    #TODO: Counts left-eye blinks within a time limit.
    def left_blink_count(self, now):
        return len([t for t in self.blinks_l if now - t <= DOUBLE_BLINK_WINDOW])

    #TODO: Detects facial gestures and returns events when the user performs actions like blinking, smiling, opening the mouth, or turning the head.
    def update(self, info, now):
        events = set()
        if info is None:                     
            keep_cd = self.head_cd
            self.reset()
            self.head_cd = keep_cd
            return events

    
        self.l_closed = info.ear_l < EAR_CLOSED or (self.l_closed and info.ear_l < EAR_OPEN)
        self.r_closed = info.ear_r < EAR_CLOSED or (self.r_closed and info.ear_r < EAR_OPEN)

        if self.both_closed:
            self.blinks_l.clear()        
            if self.both_t0 is None:
                self.both_t0 = now
            if not self.both_counted and now - self.both_t0 >= CLOSURE_MIN_SECONDS:
                self.both_counted = True      
                self.closures.append(now)
                events.add("both_closure")
        elif not self.l_closed and not self.r_closed:
            self.both_t0 = None             
            self.both_counted = False

        self.closures = [t for t in self.closures if now - t <= CLOSURE_WINDOW]
        if len(self.closures) >= CLOSURES_NEEDED:
            events.add("both_switch")
            self.closures.clear()

        l_only = self.l_closed and not self.r_closed
        r_only = self.r_closed and not self.l_closed

        self.l_frames = self.l_frames + 1 if l_only else 0
        self.r_frames = self.r_frames + 1 if r_only else 0
        if self.l_frames >= WINK_FRAMES and not self.l_latched:
            events.add("wink_left")
            self.l_latched = True           
            self.blinks_l.append(now)
        if self.r_frames >= WINK_FRAMES and not self.r_latched:
            events.add("wink_right")
            self.r_latched = True
            self.blinks_l.clear()         
        if not self.l_closed:
            self.l_latched = False
        if not self.r_closed:
            self.r_latched = False

        self.blinks_l = [t for t in self.blinks_l if now - t <= DOUBLE_BLINK_WINDOW]
        if len(self.blinks_l) >= 2:
            events.add("double_blink_left")
            self.blinks_l.clear()

        was_open = self.mouth_open
        self.mouth_open = info.mar > MAR_ON or (self.mouth_open and info.mar > MAR_OFF)
        if self.mouth_open and not was_open:
            events.add("mouth_open")

        if info.smile > SMILE_ON and info.mar < MAR_OFF:
            if self.smile_t0 is None:
                self.smile_t0 = now
            if not self.smile_latched and now - self.smile_t0 >= SMILE_HOLD:
                events.add("smile")
                self.smile_latched = True
        else:
            self.smile_t0 = None
            self.smile_latched = False

        self.yaw = 0.5 * self.yaw + 0.5 * info.yaw
        if abs(self.yaw) < HEAD_RETURN:
            new_state = 0
        elif self.yaw > HEAD_TURN:
            new_state = 1
        elif self.yaw < -HEAD_TURN:
            new_state = -1
        else:
            new_state = self.head_state
        if new_state == 0:
            self.head_armed = True
        if new_state != 0 and new_state != self.head_state and self.head_armed and now >= self.head_cd:
            events.add("turn_right" if new_state == 1 else "turn_left")
            self.head_armed = False
            self.head_cd = now + HEAD_COOLDOWN
        self.head_state = new_state
        return events


#TODO: Manages visual effects and messages displayed on the game screen.
class Feedback:
    #TODO: Initializes the lists and timer for feedback effects.
    def __init__(self):
        self.toasts = []       
        self.pops = []        
        self.flash_until = 0.0
        self.bursts = []    

    #TODO: Clears all feedback messages and visual effects.
    def clear(self):
        self.toasts, self.pops, self.bursts = [], [], []
        self.flash_until = 0.0

    #TODO: Creates a circular burst effect at a specific position.
    def burst(self, x, y, color, now):
        self.bursts.append([int(x), int(y), color, now])
        self.bursts = self.bursts[-6:]

    #TODO: Displays a temporary message on the screen.
    def toast(self, text, color, now, duration=2.0):
        self.toasts.append([text, color, now, duration])
        self.toasts = self.toasts[-2:]

    #TODO: Displays floating text that moves upward and fades away.
    def pop(self, text, x, y, color, now, duration=1.2, scale=1.0):
        self.pops.append([text, int(x), int(y), color, now, duration, scale])
        self.pops = self.pops[-6:]

    #TODO: Creates a brief white flash effect.
    def flash(self, now):
        self.flash_until = now + 0.25
    #TODO: Draws all feedback effects on the screen, including flashes, toasts, bursts, and pops.
    def draw(self, img, now):

        h, w = img.shape[:2]
        if now < self.flash_until:                 
            a = (self.flash_until - now) / 0.25
            cv2.addWeighted(img, 1 - 0.7 * a, np.full_like(img, 255), 0.7 * a, 0, dst=img)

        self.toasts = [t for t in self.toasts if now - t[2] < t[3]]
        for i, (text, color, t0, dur) in enumerate(self.toasts):
            age = now - t0
            alpha = min(1.0, age / 0.15, (dur - age) / 0.3)
            scale = fit_scale(text, 0.8, 2, w - 80)
            tw, th = text_size(text, scale, 2)
            cy = 150 + i * 56

            def paint(layer, text=text, color=color, tw=tw, th=th, cy=cy, scale=scale):
                draw_panel(layer, w / 2 - tw / 2 - 20, cy - th - 14, w / 2 + tw / 2 + 20, cy + 14, color, 0.7)
                draw_text_center(layer, text, w // 2, cy - th // 2 + 2, color, scale, 2)
            blend_draw(img, alpha, paint)

        self.bursts = [b for b in self.bursts if now - b[3] < 0.6]
        for bx, by, bcol, bt0 in self.bursts:     
            k = (now - bt0) / 0.6
            fade = 1.0 - k
            c = tuple(int(v * fade) for v in bcol)
            cv2.circle(img, (bx, by), int(20 + 70 * k), c, 3, cv2.LINE_AA)
            cv2.circle(img, (bx, by), int(10 + 40 * k), c, 2, cv2.LINE_AA)

        self.pops = [p for p in self.pops if now - p[4] < p[5]]
        for text, x, y, color, t0, dur, scale in self.pops:
            k = (now - t0) / dur                         
            fade = 1.0 - k * k
            s = scale * (1.0 + 0.35 * math.sin(min(1.0, k * 3) * math.pi))   
            c = tuple(int(v * fade) for v in color)
            draw_text_center(img, text, x, y - int(70 * k), c, s, 2)



#TODO: GAME STATE (one small class per dimension)

#TODO: Creates falling words or letters that move down the screen and sway from side to side.
class FallingItem:
    #TODO: Initializes the item's text, position, falling speed, color, and movement.
    def __init__(self, text, x, y, speed, color):
        self.text = text
        self.x = float(x)
        self.y = float(y)
        self.speed = speed
        self.color = color
        self.phase = random.uniform(0, 6.28)
        self.dx = float(x)

    #TODO: Updates the item's vertical position and side-to-side movement.
    def update(self, dt, now):
        self.y += self.speed * dt
        self.dx = self.x + math.sin(now * 2.0 + self.phase) * 12

#TODO: Finds the closest falling word or letter near the player's mouth so it can be collected.
def pick_item_near_mouth(items, mouth, reach, is_word):
    mx, my = mouth
    best, best_d = None, 1e9
    for it in items:
        if is_word:
            tw, th = text_size(it.text, 0.9, 2)
            inside = abs(mx - it.dx) <= tw / 2 + 14 + reach and abs(my - it.y) <= th / 2 + 12 + reach
        else:
            inside = math.hypot(mx - it.dx, my - it.y) <= LETTER_RADIUS + reach
        d = math.hypot(mx - it.dx, my - it.y)
        if inside and d < best_d:
            best, best_d = it, d
    return best

#TODO: Stores the items and collected letters for Dimension 2.
class Dim2State:

    #TODO: Initializes the Dimension 2 state.
    def __init__(self):
        self.reset()
    #TODO: Resets the items, collected letters, and timers.
    def reset(self):
        self.items = []
        self.collected = []
        self.spawn_t = 0.0
        self.collect_cd = 0.0

#TODO: Manages the game state for Dimension 3, including lives, levels, words, and missing letters.
class Dim3State:
    #TODO: Initializes the Dimension 3 state.
    def __init__(self):
        self.reset()

    #TODO: Resets the game state, restores lives, and starts the first level.
    def reset(self):
        self.lives = MAX_LIVES
        self.phase = "play"     
        self.phase_until = 0.0
        self.banner_title = ""
        self.banner_sub = ""
        self.spawn_t = 0.0
        self.collect_cd = 0.0
        self.recent = []           
        self.letters = []
        self.start_level(0)
    #TODO: Starts a new level and resets the word progress.
    def start_level(self, level):
        self.level = level
        self.words_done = 0
        self.phase = "play"
        self.load_word()
    #TODO: Selects a word, hides some letters, and avoids recently used words when possible.
    def load_word(self):
        length, missing = level_spec(self.level)
        pool = words_for_length(length)
        keep = min(len(pool) - 1, 10)                
        banned = set(self.recent[-keep:]) if keep > 0 else set()
        choices = [w for w in pool if w not in banned] or pool
        word = random.choice(choices)
        self.recent.append(word)
        self.recent = self.recent[-30:]
        self.word = word.upper()
        missing = min(missing, len(self.word) - 1)
        self.hidden = set(random.sample(range(len(self.word)), missing))  
        self.filled = set()
        self.letters = []
    #TODO: Returns the missing letters that the player still needs to collect.
    def needed_letters(self):
        return sorted({self.word[i] for i in self.hidden - self.filled})



#TODO:  GAME LOGIC

#TODO: pawns falling words, updates their movement, removes words that leave the screen,
#TODO: and lets the player collect words by opening their mouth. It also checks if the word bubble is full and
#TODO: displays visual effects when a word is collected.
def update_dim2(st, dt, now, info, tracker, fb, w, h):
    if now >= st.spawn_t and len(st.items) < 7:
        on_screen = {it.text for it in st.items}
        choices = [wd.upper() for wd in DIM2_WORDS if wd.upper() not in on_screen] or [wd.upper() for wd in DIM2_WORDS]
        st.items.append(FallingItem(
            random.choice(choices), random.randint(110, max(111, w - 110)), 60,
            random.uniform(90, 170) * h / 720.0, random.choice(FALL_COLORS)))
        st.spawn_t = now + random.uniform(1.0, 1.8)

    for it in st.items:
        it.update(dt, now)
    st.items = [it for it in st.items if it.y < h + 40]

    if info is not None and tracker.mouth_open and now >= st.collect_cd:
        hit = pick_item_near_mouth(st.items, info.mouth, info.face_w * 0.2, True)
        if hit is not None:
            if len(st.collected) >= MAX_COLLECTED_WORDS:
                fb.toast("Bubble is full! Blink your LEFT eye twice to remove a word.", BAD_COLOR, now)
                st.collect_cd = now + 1.5
            else:
                st.items.remove(hit)
                st.collected.append(hit.text)
                st.collect_cd = now + COLLECT_COOLDOWN
                fb.pop(random.choice(WORD_PRAISE) + f"  +{hit.text}", hit.dx, hit.y, GOOD_COLOR, now, 1.2, 0.9)
                fb.burst(hit.dx, hit.y, GOOD_COLOR, now)

#TODO: Spawns a falling letter, prioritizing missing letters needed to complete the current word, and sets its position, speed, and color.
def spawn_letter(st, now, w, h):
    needed = st.needed_letters()
    on_screen = {it.text for it in st.letters}
    if needed and (not any(n in on_screen for n in needed) or random.random() < 0.3):
        ch = random.choice(needed)        
    else:
        ch = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    speed = (110 + min(st.level, 40) * 16) * h / 720.0      
    st.letters.append(FallingItem(ch, random.randint(60, max(61, w - 60)), 150, speed, random.choice(FALL_COLORS)))

#TODO: Updates Dimension 3 by spawning falling letters, allowing the player to collect missing letters by opening their mouth,
#TODO: reducing lives for wrong letters, and advancing to the next level when the word is completed.
def update_dim3(st, dt, now, info, tracker, fb, w, h):

    if st.phase != "play":                          
        if now >= st.phase_until:
            if st.phase == "game_over":
                st.reset()                         
                fb.clear()
            else:                                    
                st.start_level(st.level + 1)
        return

    if now >= st.spawn_t and len(st.letters) < MAX_LETTERS_ON_SCREEN:
        spawn_letter(st, now, w, h)
        st.spawn_t = now + max(0.12, 0.34 - min(st.level, 20) * 0.011)

    for it in st.letters:
        it.update(dt, now)
    st.letters = [it for it in st.letters if it.y < h + 40]

    if info is None or not tracker.mouth_open or now < st.collect_cd:
        return
    hit = pick_item_near_mouth(st.letters, info.mouth, info.face_w * 0.2, False)
    if hit is None:
        return
    st.letters.remove(hit)
    st.collect_cd = now + COLLECT_COOLDOWN
    slot = next((i for i in sorted(st.hidden - st.filled) if st.word[i] == hit.text), None)

    if slot is None:                               
        st.lives -= 1
        fb.burst(hit.dx, hit.y, BAD_COLOR, now)
        if st.lives <= 0:
            st.letters = []
            st.phase = "game_over"
            st.banner_title = "GAME OVER"
            st.banner_sub = f"The word was {st.word}  -  back to Level 1"
            st.phase_until = now + 3.0
        else:
            fb.pop(f"Wrong! {st.lives} chance{'s' if st.lives != 1 else ''} left", hit.dx, hit.y, BAD_COLOR, now, 1.3, 0.8)
        return

    st.filled.add(slot)                       
    fb.pop(random.choice(GOOD_MESSAGES), hit.dx, hit.y, GOOD_COLOR, now, 1.3, 0.9)
    fb.burst(hit.dx, hit.y, GOOD_COLOR, now)

    if st.filled == st.hidden:                     
        st.letters = []
        st.words_done = 1
        st.phase = "level_done"
        st.banner_title = "CORRECT WORD!"
        st.banner_sub = st.word
        st.phase_until = now + 1.8

#TODO: Draws the optional face mesh on the camera frame, then blends the overlay with the original frame.
def draw_faces(frame, infos, theme, frame_count, show_mesh):
    overlay = frame.copy()
    for info in infos:
        color = theme["primary"]
        if show_mesh:
            mp_drawing.draw_landmarks(
                image=overlay, landmark_list=info.landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=theme["mesh"], thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(
                image=overlay, landmark_list=info.landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1))
    return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

#TODO: Splits a list of words into multiple lines so the text fits within the available screen width.
def wrap_words(words, max_w, scale, thickness):
    lines, cur = [], ""
    for wd in words:
        test = (cur + " " + wd).strip()
        if cur and text_size(test, scale, thickness)[0] > max_w:
            lines.append(cur)
            cur = wd
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines

#TODO: Draws a thought bubble above the player's head to display collected words, adjusting the text size and bubble position to fit the screen.
def draw_thought_bubble(img, words, head_x, head_top_y, color):
    h, w = img.shape[:2]
    scale, thick = 0.9, 2
    max_w = min(380, w - 160)
    lines = wrap_words(words, max_w, scale, thick) if words else ["..."]
    if len(lines) > 4:
        scale = 0.7
        lines = wrap_words(words, max_w, scale, thick)
    line_h = int(text_size("Ag", scale, thick)[1] * 1.9)
    block_w = max(text_size(l, scale, thick)[0] for l in lines)
    block_h = len(lines) * line_h
    rx = int(block_w / 2 * 1.25) + 34
    ry = int(block_h / 2 * 1.25) + 28

    cx = int(min(max(head_x, rx + 15), w - rx - 15))
    cy = int(max(head_top_y - 70 - ry, 50 + ry))      

    #TODO: Draws a cloud-shaped thought bubble using an ellipse and circles, with adjustable size and color.
    def cloud(layer, grow, col):
        cv2.ellipse(layer, (cx, cy), (rx + grow, ry + grow), 0, 0, 360, col, -1, cv2.LINE_AA)
        for a in range(0, 360, 30):                 
            bx = int(cx + (rx - 4) * math.cos(math.radians(a)))
            by = int(cy + (ry - 4) * math.sin(math.radians(a)))
            cv2.circle(layer, (bx, by), int(ry * 0.22) + 8 + grow, col, -1, cv2.LINE_AA)

    #TODO: Draws the thought bubble, adds small circles connecting it to the player's head, and displays the collected words inside the bubble.
    def paint(layer):
        cloud(layer, 3, color)                       
        cloud(layer, 0, (250, 248, 255))              
        start = (cx, cy + ry + 14)
        end = (int(head_x), int(head_top_y) - 6)
        if end[1] - start[1] > 20:                    
            for t, r in ((0.15, 13), (0.5, 9), (0.85, 6)):
                px = int(start[0] + (end[0] - start[0]) * t)
                py = int(start[1] + (end[1] - start[1]) * t)
                cv2.circle(layer, (px, py), r + 3, color, -1, cv2.LINE_AA)
                cv2.circle(layer, (px, py), r, (250, 248, 255), -1, cv2.LINE_AA)
    blend_draw(img, 0.93, paint)

    y = cy - block_h // 2 + line_h - (line_h - text_size("A", scale, thick)[1]) // 2
    for l in lines:
        tw = text_size(l, scale, thick)[0]
        txt_col = (60, 30, 80) if words else (160, 150, 170)
        cv2.putText(img, l, (cx - tw // 2, y - 4), FONT, scale, txt_col, thick, cv2.LINE_AA)
        y += line_h

#TODO: Draws a falling word inside a colored panel on the game screen.
def draw_falling_word(img, it):
    scale, th = 0.9, 2
    tw, tht = text_size(it.text, scale, th)
    x, y = int(it.dx), int(it.y)
    draw_panel(img, x - tw // 2 - 14, y - tht // 2 - 12, x + tw // 2 + 14, y + tht // 2 + 12, it.color, 0.55)
    draw_text_center(img, it.text, x, y, WHITE, scale, th)

#TODO: Draws a falling letter inside a circular design with a colored outline.
def draw_falling_letter(img, it):
    x, y = int(it.dx), int(it.y)
    cv2.circle(img, (x, y), LETTER_RADIUS + 4, it.color, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), LETTER_RADIUS, (35, 20, 45), -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), LETTER_RADIUS, it.color, 2, cv2.LINE_AA)
    draw_text_center(img, it.text, x, y, WHITE, 0.75, 2)

#TODO: Draws a circle around the player's mouth, changing its color and pulsing when the mouth is open to show where to collect falling words or letters.
def draw_mouth_target(img, info, tracker, now):
    color = GOOD_COLOR if tracker.mouth_open else (200, 200, 200)
    r = int(info.face_w * 0.2) + (int(4 * math.sin(now * 8)) if tracker.mouth_open else 0)
    cv2.circle(img, info.mouth, max(10, r), color, 2, cv2.LINE_AA)

#TODO: Displays the gesture controls on the screen, showing progress toward switching dimensions or undoing a collected word using eye blinks.
def draw_gesture_panel(img, app, tracker, now, theme):
    h, w = img.shape[:2]
    if app.dim == 0:
        rows = [("BOTH EYES x6 > DIM 2", tracker.closure_count(now), CLOSURES_NEEDED),
                ("L-EYE x2 > UNDO WORD", tracker.left_blink_count(now), 2)]
    else:
        rows = [("BOTH EYES x6 > DIM 1", tracker.closure_count(now), CLOSURES_NEEDED)]
    x1, x2 = w - 292, w - 12
    y1 = 122 if app.dim == 1 else 50          
    y2 = y1 + 14 + 26 * len(rows)
    draw_panel(img, x1, y1, x2, y2, theme["primary"], 0.6)
    for i, (label, val, total) in enumerate(rows):
        y = y1 + 26 + i * 26
        draw_hud_text(img, label, (x1 + 12, y), theme["accent"], 0.42, 1)
        for k in range(total):
            cx = x2 - 14 - (total - 1 - k) * 15
            on = k < val
            cv2.circle(img, (cx, y - 5), 6, theme["primary"] if on else (110, 110, 110),
                       -1 if on else 2, cv2.LINE_AA)

#TODO: Displays Dimension 1's filter controls, including the current filter name, filter number, navigation arrows, and indicator dots.
def draw_dim1(img, app, theme, now, tracker):
    h, w = img.shape[:2]
    cy = h - 160
    draw_panel(img, w / 2 - 300, cy - 55, w / 2 + 300, cy + 55, theme["primary"], 0.6)
    draw_text_center(img, f"FILTER {app.filter_idx + 1} / {NUM_FILTERS}", w // 2, cy - 32, theme["accent"], 0.55, 1)
    label = GLASS_STYLES[app.filter_idx][0] if app.filter_on else "FILTER OFF  (press X)"
    draw_text_center(img, label, w // 2, cy + 2, WHITE, 1.0 if app.filter_on else 0.8, 2)
    wob = int(6 * math.sin(now * 6))
    draw_text_center(img, "<", w // 2 - 270 - wob, cy, theme["primary"], 1.4, 3)
    draw_text_center(img, ">", w // 2 + 270 + wob, cy, theme["primary"], 1.4, 3)
    for i in range(NUM_FILTERS):                            
        x = int(w / 2 + (i - (NUM_FILTERS - 1) / 2) * 24)
        on = i == app.filter_idx
        cv2.circle(img, (x, cy + 38), 6 if on else 4, theme["primary"] if on else (120, 120, 120), -1, cv2.LINE_AA)

#TODO: Draws the falling words, displays the mouth target, and shows how many words the player has collected.
def draw_dim2(img, st, info, tracker, now, theme):
    for it in st.items:
        draw_falling_word(img, it)
    if info is not None:
        draw_mouth_target(img, info, tracker, now)
    draw_hud_text(img, f"WORDS: {len(st.collected)}/{MAX_COLLECTED_WORDS}", (12, 70), theme["accent"], 0.6, 2)


#TODO: Displays the missing-letter word puzzle, current level, remaining chances, progress bar,
#TODO: falling letters, mouth target, and level-complete or game-over messages.
def draw_dim3(img, st, info, tracker, now, theme):
    h, w = img.shape[:2]
    length, missing = level_spec(st.level)

    n = len(st.word)
    slot = int(min(54, (w - 560) / max(1, n) - 8))
    slot = max(24, slot)
    total = n * (slot + 8) - 8
    x0, y0 = int(w / 2 - total / 2), 52
    for i, ch in enumerate(st.word):
        x = x0 + i * (slot + 8)
        if i in st.hidden and i not in st.filled:
            border, label, tcol = GOLD_COLOR, "_", GOLD_COLOR
            if int(now * 3) % 2 == 0:
                border = theme["accent"]
        elif i in st.hidden:
            border, label, tcol = GOOD_COLOR, ch, GOOD_COLOR
        else:
            border, label, tcol = theme["primary"], ch, WHITE
        draw_panel(img, x, y0, x + slot, y0 + slot + 6, border, 0.65)
        draw_text_center(img, label, x + slot // 2, y0 + slot // 2 + 3, tcol, slot / 40.0, 2)

   
    draw_panel(img, 12, 52, 250, 172, theme["primary"], 0.6)
    draw_hud_text(img, f"LEVEL {st.level + 1}", (24, 78), theme["primary"], 0.65, 2)
    draw_hud_text(img, f"LETTERS: {length}   MISSING: {missing}", (24, 102), WHITE, 0.5)
    draw_hud_text(img, "CHANCES:", (24, 128), GOLD_COLOR, 0.5, 1)
    for k in range(MAX_LIVES):                              
        cx = 128 + k * 26
        if k < st.lives:
            cv2.circle(img, (cx, 123), 9, BAD_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, 123), 9, BAD_COLOR, 2, cv2.LINE_AA)
    frac = len(st.filled) / max(1, len(st.hidden))             
    cv2.rectangle(img, (24, 146), (236, 156), (50, 50, 50), -1)
    cv2.rectangle(img, (24, 146), (24 + int(212 * frac), 156), theme["accent"], -1)

    for it in st.letters:
        draw_falling_letter(img, it)
    if info is not None:
        draw_mouth_target(img, info, tracker, now)

    if st.phase != "play":                                   
        col = BAD_COLOR if st.phase == "game_over" else GOOD_COLOR
        tw = max(text_size(st.banner_title, 1.2, 3)[0], text_size(st.banner_sub, 0.8, 2)[0])
        extra = 34 if st.phase == "game_over" else 0
        draw_panel(img, w / 2 - tw / 2 - 40, h / 2 - 75, w / 2 + tw / 2 + 40, h / 2 + 55 + extra, col, 0.75)
        draw_text_center(img, st.banner_title, w // 2, h // 2 - 28, col, 1.2, 3)
        draw_text_center(img, st.banner_sub, w // 2, h // 2 + 18, WHITE, 0.8, 2)
        if st.phase == "game_over":                         
            left = max(1, int(math.ceil(st.phase_until - now)))
            draw_text_center(img, f"Restarting in {left}...", w // 2, h // 2 + 56, GOLD_COLOR, 0.8, 2)

#TODO: Stores the short gesture instructions displayed for each game dimension.
INSTRUCTIONS = [
    "Head LEFT/RIGHT: filter | Mouth: collect word | L-eye x2: undo word | BOTH eyes 6x: Dim 2 | Smile: photo",
    "Mouth: catch letters (3 wrong = game over) | BOTH eyes 6x: Dim 1 | X: filter | Smile: photo",
]

#TODO: Stores the detailed help messages explaining the game controls, screenshots, filters, gestures, and reset rules.
HELP_LINES = [
    ("EITHER DIM", "SMILE = automatic photo (saved in screenshots/), with your current filter"),
    ("  Q / ESC", "quit"),
    ("  S", "save manual screenshot to screenshots/"),
    ("  X", "turn the sunglasses / face filter on or off"),
    ("  F", "re-focus window"),
    ("  H / M", "show-hide this help / toggle face mesh"),
    ("DIMENSION 1", "head RIGHT / LEFT = next / previous filter, mouth = collect falling word"),
    ("", "LEFT eye blink x2 = remove last word, close BOTH eyes 6 separate times = Dim 2"),
    ("DIMENSION 2", "mouth = catch letter (3 wrong = game over), BOTH eyes 6 separate times = Dim 1"),
    ("RESET RULE", "leaving a dimension resets it; Dim 2 always restarts at Level 1 (also after game over)"),
]

#TODO: Displays the help panel on the screen, showing game instructions and controls in organized rows, including keyboard keys and facial gestures.
def draw_help(img, theme):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = 40, 60, min(w - 40, 1120), 60 + 50 + len(HELP_LINES) * 30
    draw_panel(img, x1, y1, x2, y2, theme["primary"], 0.75)
    draw_hud_text(img, "HELP  (press H to hide)", (x1 + 20, y1 + 32), theme["primary"], 0.8, 2)
    for i, (a, b) in enumerate(HELP_LINES):
        y = y1 + 70 + i * 30
        draw_hud_text(img, a, (x1 + 20, y), theme["accent"], 0.55, 1)
        draw_hud_text(img, b, (x1 + 200, y), (235, 235, 235), 0.5, 1)


#TODO: Displays the game's heads-up display (HUD), including the current dimension, filter name, instructions, FPS, face count, detected gestures, and keyboard shortcuts.
def draw_hud(frame, app, theme, fps, face_count, now):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    draw_hud_text(frame, DIMENSION_TITLES[app.dim], (12, 26), theme["primary"], 0.65, 2)
    if app.filter_on:
        ftxt = f"FILTER {app.filter_idx + 1}/{NUM_FILTERS}: {GLASS_STYLES[app.filter_idx][0]}"
    else:
        ftxt = "FILTER: OFF  (X)"
    draw_hud_text(frame, ftxt, (w - text_size(ftxt, 0.55, 1)[0] - 16, 26), theme["accent"], 0.55)

    text = INSTRUCTIONS[app.dim]
    scale = fit_scale(text, 0.55, 1, w - 60)
    tw, _ = text_size(text, scale, 1)
    draw_panel(frame, w / 2 - tw / 2 - 16, h - 78, w / 2 + tw / 2 + 16, h - 44, theme["primary"], 0.6)
    draw_text_center(frame, text, w // 2, h - 61, WHITE, scale, 1)

    cv2.rectangle(frame, (0, h - 34), (w, h), (0, 0, 0), -1)
    draw_hud_text(frame, f"FPS: {fps:4.1f}", (12, h - 12), theme["accent"])
    draw_hud_text(frame, f"FACES: {face_count}", (130, h - 12), theme["accent"])
    if app.gesture and now - app.gesture_t < 1.5:           
        draw_hud_text(frame, f"GESTURE: {app.gesture}", (260, h - 12), GOLD_COLOR, 0.55, 1)
    draw_hud_text(frame, "[H] help [M] mesh [S] shot [X] filter [F] focus [Q] quit",
                  (w - 470, h - 12), (200, 200, 200), 0.45)





#TODO: Stores the names and display labels for detected facial gestures, such as blinking, smiling, opening the mouth, and turning the head.
GESTURE_LABELS = [
    ("both_switch", "BOTH EYES x6 - SWITCHING"),
    ("double_blink_left", "LEFT EYE DOUBLE BLINK"),
    ("smile", "SMILE"),
    ("turn_right", "HEAD RIGHT"),
    ("turn_left", "HEAD LEFT"),
    ("both_closure", "BOTH EYES CLOSED"),
    ("wink_left", "LEFT EYE BLINK"),
    ("wink_right", "RIGHT EYE BLINK"),
    ("mouth_open", "MOUTH OPEN"),
]

#TODO: Stores the game's main settings and current state, including the active dimension, sunglasses filter, help panel, face mesh, and detected gestures.
class App:
    def __init__(self):
        self.dim = 0              
        self.filter_idx = 0       
        self.filter_on = True        
        self.lock_until = 0.0
        self.smile_cd = 0.0
        self.show_help = True
        self.show_mesh = False    
        self.eye_smooth = {}       
        self.head_pos = None         
        self.head_seen = 0.0
        self.gesture = ""
        self.gesture_t = 0.0

#TODO: Switches to the selected game dimension, resets the game states and feedback, adds a short switching delay, and displays a message showing the new dimension.
def goto_dimension(app, idx, now, tracker, d2, d3, fb, w, h):
    app.dim = idx
    d2.reset()
    d3.reset()
    fb.clear()                                   
    app.lock_until = now + LOCK_AFTER_SWITCH      
    tracker.reset(keep_smile=True)                 
    fb.pop(DIMENSION_TITLES[idx], w // 2, h // 2 - 60, THEME["primary"], now, 1.6, 1.3)



#TODO:  MAIN

#TODO: Opens the webcam or selected camera source and tries another method if the first attempt fails.
def open_camera():
    if isinstance(CAM_SOURCE, int):
        cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(CAM_SOURCE)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAM_SOURCE)
    else:
        cap = cv2.VideoCapture(CAM_SOURCE)      
    return cap

#TODO: Starts the game by opening the webcam, initializing the game components, detecting faces, and preparing the camera frames for processing.
def main():
    cap = open_camera()
    if not cap.isOpened():
        print("ERROR: Could not open the webcam.")
        print("  - Check that no other app is using the camera.")
        print("  - Check camera permission for Python / your terminal.")
        print("  - Or change CAM_SOURCE at the top of this file.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    app = App()
    fb = Feedback()
    d2, d3 = Dim2State(), Dim3State()
    tracker = GestureTracker()
    theme = THEME                                
    frame_count = 0
    fail_count = 0
    prev_time = time.time()
    fps = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            while True:
                ok, raw = cap.read()
                if not ok:                         
                    fail_count += 1
                    if fail_count > 60:
                        print("ERROR: Lost the webcam feed.")
                        break
                    cv2.waitKey(10)
                    continue
                fail_count = 0

                raw = cv2.flip(raw, 1)             
                h, w = raw.shape[:2]
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)

                infos = []
                if results.multi_face_landmarks:
                    infos = [analyze_face(lm, w, h) for lm in results.multi_face_landmarks]
                primary = max(infos, key=lambda i: i.area) if infos else None
                by_x = sorted(infos, key=lambda i: i.cx)[:2]   

                now = time.time()
                dt = max(0.0, min(0.1, now - prev_time))

                #TODO: This code detects facial gestures, temporarily disables them after switching dimensions, and displays the detected gesture label on the screen.
                events = tracker.update(primary, now)
                if now < app.lock_until:
                    events = set()              

                for key_name, label in GESTURE_LABELS:    
                    if key_name in events:
                        app.gesture, app.gesture_t = label, now
                        break

                #TODO: This code tracks the player's head position, smoothing it over time, and updates the head position for displaying the thought bubble with collected words.
                if primary is not None:
                    hx, hy = primary.head_top
                    if app.head_pos is None:
                        app.head_pos = (hx, hy)
                    else:
                        app.head_pos = (app.head_pos[0] * 0.65 + hx * 0.35, app.head_pos[1] * 0.65 + hy * 0.35)
                    app.head_seen = now

                #TODO: This code switches between game dimensions, changes filters with head turns, collects or removes words,
                #TODO: draws the game elements and facial effects, captures photos when smiling, displays the game window, and
                #TODO: handles keyboard controls like quit, screenshot, filter, help, and face mesh.
                def go(idx):
                    goto_dimension(app, idx, now, tracker, d2, d3, fb, w, h)

                turn_l, turn_r = "turn_left" in events, "turn_right" in events
                both_n = "both_switch" in events                  
                double_l = "double_blink_left" in events

                if app.dim == 0:                   
                    if both_n:
                        go(1)                    
                    else:
                        if turn_l or turn_r:      
                            app.filter_idx = (app.filter_idx + (1 if turn_r else -1)) % NUM_FILTERS
                            app.filter_on = True
                            fb.pop(GLASS_STYLES[app.filter_idx][0], w // 2, h // 3, theme["accent"], now, 1.0, 1.4)
                        if double_l:             
                            if d2.collected:
                                fb.toast(f'Removed "{d2.collected.pop()}"', BAD_COLOR, now, 1.5)
                        update_dim2(d2, dt, now, primary, tracker, fb, w, h)

                else:                          
                    if both_n:
                        go(0)                     
                    else:
                        update_dim3(d3, dt, now, primary, tracker, fb, w, h)

                frame = raw
                app.eye_smooth = {k: v for k, v in app.eye_smooth.items() if k < len(by_x)}
                glass_art = get_glasses(app.filter_idx) if app.filter_on else None
                if glass_art is not None:
                    for slot, info in enumerate(by_x):
                        eye_l, eye_r = smooth_eyes(app, slot, info)
                        draw_glasses(frame, eye_l, eye_r, glass_art)

                take_photo = "smile" in events and now >= app.smile_cd
                if take_photo:
                    app.smile_cd = now + SMILE_PHOTO_COOLDOWN

                if app.dim == 0 and app.head_pos is not None and now - app.head_seen < 0.6:
                    draw_thought_bubble(frame, d2.collected, app.head_pos[0], app.head_pos[1], theme["primary"])

                frame = draw_faces(frame, infos, theme, frame_count, app.show_mesh)

                if app.dim == 0:
                    draw_dim1(frame, app, theme, now, tracker)
                    draw_dim2(frame, d2, primary, tracker, now, theme)
                else:
                    draw_dim3(frame, d3, primary, tracker, now, theme)
                draw_gesture_panel(frame, app, tracker, now, theme)

                if not infos:
                    if int(now * 2) % 2 == 0:
                        draw_text_center(frame, "NO FACE DETECTED - step into the camera view",
                                         w // 2, h // 2, GOLD_COLOR, 0.8, 2)

                fb.draw(frame, now)

                dt_fps = now - prev_time
                prev_time = now
                if dt_fps > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt_fps)

                draw_hud(frame, app, theme, fps, len(infos), now)

                if take_photo:
                    path = save_image(frame, "smile")
                    fb.flash(now)              
                    if path:
                        fb.toast("Smile detected! Photo captured!", GOOD_COLOR, now, 2.5)
                    else:
                        fb.toast("Could not save the photo!", BAD_COLOR, now, 2.5)

                if app.show_help:
                    draw_help(frame, theme)

                cv2.imshow(WINDOW_NAME, frame)

                if frame_count == 5:
                    force_window_focus(WINDOW_NAME)
                frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    path = save_image(frame, "screenshot")
                    fb.toast(f"Screenshot saved: {path}" if path else "Could not save screenshot",
                             GOOD_COLOR if path else BAD_COLOR, now, 2.0)
                elif key == ord("x"):
                    app.filter_on = not app.filter_on
                    if app.filter_on and GLASS_STYLES[app.filter_idx][1] is None:
                        app.filter_idx = 0
                    fb.toast("Filter ON" if app.filter_on else "Filter OFF - normal face",
                             GOOD_COLOR if app.filter_on else GOLD_COLOR, now, 1.5)
                elif key == ord("f"):
                    force_window_focus(WINDOW_NAME)
                elif key == ord("h"):
                    app.show_help = not app.show_help
                elif key == ord("m"):
                    app.show_mesh = not app.show_mesh
                elif app.dim == 0 and key in (ord("u"), 8):
                    if d2.collected:
                        fb.toast(f'Removed "{d2.collected.pop()}"', BAD_COLOR, now, 1.5)

                try:
                    if frame_count > 10 and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()