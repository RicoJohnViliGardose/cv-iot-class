"""
AR FACE TRACKER - a classroom demo.

Real-time face tracking with a sci-fi HUD overlay: a glowing face mesh,
a targeting-reticle bounding box, and a live FPS counter.

Controls:
    M - toggle mesh overlay
    B - toggle bounding box / reticle
    E - toggle mood detection (off by default)
    R - recalibrate mood baseline (hold a neutral face when you press it)
    C - cycle HUD color theme
    S - save a screenshot
    F - re-focus the window (use this if keypresses stop responding)
    Q / ESC - quit
"""

import ctypes
import math
import time

import cv2
import mediapipe as mp


#TODO: Config


WINDOW_NAME = "AR FACE TRACKER"

CAM_SOURCE = 0
#TODO: Laptop with a build- in webcam: leave as the drive index above (0).
#TODO: Lab PC with no webcam: set to your phone's IP Webcam stream tunneled over
#TODO: USB via 'asb forward tcp:8080 tcp:8080 ', e.g.:
#TODO: CAM_SOURCE = "http://localhost:8080/video"

THEMES = [
    {"name": "CYAN", "primary": (255, 255, 0), "accent": (0, 255, 255), "mesh": (255, 200, 0),},
    {"name": "MAGENTA", "primary": (255, 0, 255), "accent": (180, 0, 255), "mesh": (255, 100, 255),},
    {"name": "GREEN", "primary": (0, 255, 100), "accent": (0, 255, 0), "mesh": (0, 255, 150),},
    {"name": "AMBER", "primary": (0, 200, 255), "accent": (0, 165, 255), "mesh": (0, 220, 255),},
]

BRACKET_LEN = 30
BRACKET_THICKNESS = 3

#TODO: MediaPipe face Mesh landmark indices used for the mood heuristic.
LM_LEFT_EYE_OUTER = 33
LM_RIGHT_EYE_OUTER = 263
LM_MOUSE_LEFT = 61
LM_MOUSE_RIGHT = 291
LM_MOUSE_TOP = 13
LM_MOUSE_BOTTOM = 14

#TODO: Mouth-openness threshold for SURPRISED is absoulute (ratio to inter-eye
#TODO: distance), since a neutral mouth is essentially always closed. HAPPY/SAD
#TODO: instead compare against a per-person calibrated neutral baseline (see
#TODO: mood_calibration below) because resting mouth width/shape varies a lot
#TODO: from face to face - a fixed threshold under - or over-fires depending on 
#TODO: who's in frame.
MOOD_OPEN_MOUTH_RATIO = 0.22
MOOD_CALIBRATION_FRAMES = 20
MOOD_HAPPY_LIFT_DELTA = 0.035
MOOD_HAPPY_WIDTH_DELTA = 0.04
MOOD_SAD_LIFT_DELTA = -0.025

MOOD_COLORS = {
    "HAPPY": (0, 220, 0),
    "SURPRISED": (0, 165, 255),
    "SAD": (255, 120, 0),
    "NEUTRAL": (200, 200, 200),
}


#TODO: Helpers



def draw_glow_line(img, pt1, pt2, color, thickness=1):
    """Draw a line with a soft outer glow to sell the 'AR' look."""
    cv2.line(img, pt1, pt2, color, thickness + 3, cv2.LINE_AA)
    cv2.line(img, pt1, pt2, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)


def draw_corner_brackets(img, x1, y1, x2, y2, color):
    """Draw targeting-reticle style corner brackets instead of a plain box."""
    corners = [
        ((x1, y1), (1,0), (0,1)),
        ((x2, y1), (-1,0), (0,1)),
        ((x1, y2), (1,0), (0,-1)),
        ((x2, y2), (-1,0), (0,-1)),
    ]
    for (cx, cy), dx, dy in corners:
        p1 = (cx + dx[0] * BRACKET_LEN, cy)
        p2 = (cx, cy)
        p3 = (cx, cy + dy[1] * BRACKET_LEN)
        draw_glow_line(img, p1, p2, color, BRACKET_THICKNESS)
        draw_glow_line(img, p2, p3, color, BRACKET_THICKNESS)


def draw_hud_text(img, text, org, color, scale=0.55, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def force_window_focus(window_title):
    """Windows blocks background proceeses from silently stealing focus, so a
    freashly- opened cv2 window sometimes never receives keyboard input even
    after being clicked. Explicitly activate it once via the Win32 API."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if hwnd:
            user32.ShowWindow(hwnd, 5)
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def mouth_metrics(landmarks, img_w, img_h):
    """Mouth width, mouth height, and corner lift, all normalized by
    inter-eye distance so they hold up regardless of how close a face is 
    to the camera. Returns None if the eyes are degenerately close 
    (shouldn't happen with a real face)."""

    def pt(idx):
        lm = landmarks.landmark[idx]
        return lm.x * img_w, lm.y * img_h
    
    left_eye = pt(LM_LEFT_EYE_OUTER)
    right_eye = pt(LM_RIGHT_EYE_OUTER)
    interocular = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    if interocular < 1e-3:
        return None

    mouth_left = pt(LM_MOUSE_LEFT)
    mouth_right = pt(LM_MOUSE_RIGHT)
    mouth_top = pt(LM_MOUSE_TOP)
    mouth_bottom = pt(LM_MOUSE_BOTTOM)

    mouth_width = math.hypot(mouth_right[0] - mouth_left[0], mouth_right[1] - mouth_left[1]) / interocular
    mouth_height = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1]) / interocular

    mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
    corner_avg_y = (mouth_left[1] + mouth_right[1]) / 2
    mouth_lift = (mouth_center_y - corner_avg_y) / interocular

    return mouth_width, mouth_height, mouth_lift


def classify_mood(metrics, baseline):
    """Classify mood from mouth_metris() output. 'baseline' is the
    (mouth_width, corner_lift) measured during calibration on this
    person's neutral face - HAPPY/SAD are relative to that, not absoulte,
    since resting mouth shape varies alot between faces."""
    mouth_width, mouth_height, corner_lift = metrics
    if mouth_height > MOOD_OPEN_MOUTH_RATIO:
        return "SURPRISED"

    based_width, based_lift = baseline
    lift_delta = corner_lift - based_lift
    width_delta = mouth_width - based_width

    if lift_delta > MOOD_HAPPY_LIFT_DELTA and width_delta > MOOD_HAPPY_WIDTH_DELTA:
        return "HAPPY"
    if lift_delta < MOOD_SAD_LIFT_DELTA:
        return "SAD"
    return "NEUTRAL"


def landmarks_bounding_box(landmarks, img_w, img_h):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    return(
        int(min(xs) * img_w),
        int(min(ys) * img_h),
        int(max(xs) * img_w),
        int(max(ys) * img_h)
    )


def draw_scan_line(img, frame_count, color, top, bottom, left, right):
    """An animated horizontal scan line sweeping throught the face box."""
    span = bottom - top
    if span <= 0:
        return
    offset = frame_count % 60
    y = top + int((offset / 60) * span)
    cv2.line(img, (left, y), (right, y), color, 1, cv2.LINE_AA)



#TODO: Main



def main():
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    if isinstance(CAM_SOURCE, int):
        cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAM_SOURCE)
    else:
        cap = cv2.VideoCapture(CAM_SOURCE) #TODO: URL source: let OpenCV pick the FFmpeg backed
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam. Check CAM_SOURCE / camera permission.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    show_mesh = True
    show_box = True
    show_mood = False
    theme_idx = 0
    frame_count = 0
    shot_count = 0
    prev_time = time.time()
    fps = 0.0

    mood_calibrating = False
    mood_calib_count = 0
    mood_calib_sum_width = 0.0
    mood_calib_sum_lift = 0.0
    mood_baseline = (0.6, 0.0)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh, mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.6
    ) as face_detection:
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            mesh_results = face_mesh.process(rgb)
            det_results = face_detection.process(rgb)

            theme = THEMES[theme_idx]
            overlay = frame.copy()
            face_count = 0

            if det_results.detections:
                face_count = len(det_results.detections)
                for det in det_results.detections:
                    box = det.location_data.relative_bounding_box
                    x1 = max(0, int(box.xmin * w) - 15)
                    y1 = max(0, int(box.ymin * h) - 15)
                    x2 = min(w, int((box.xmin + box.width) * w) + 15)
                    y2 = min(h, int((box.ymin + box.height) * h) + 15)

                    if show_box:
                        draw_corner_brackets(overlay, x1, y1, x2, y2, theme["primary"])
                        draw_scan_line(overlay, frame_count, theme["accent"], y1, y2, x1, x2)
                        conf = det.score[0] if det.score else 0.0
                        draw_hud_text(
                            overlay, f"FACE LOCK {conf * 100:4.1f}%",
                            (x1, max(20, y1 - 12)), theme["primary"], 0.5,
                        )

            if mesh_results.multi_face_landmarks:
                if show_mood and mood_calibrating:
                    metrics = mouth_metrics(mesh_results.multi_face_landmarks[0], w, h)
                    if metrics is not None:
                        mood_calib_sum_width += metrics[0]
                        mood_calib_sum_lift += metrics[2]
                        mood_calib_count += 1
                        if mood_calib_count >= MOOD_CALIBRATION_FRAMES:
                            mood_baseline = (
                                mood_calib_sum_width / mood_calib_count,
                                mood_calib_sum_lift / mood_calib_count
                            )
                            mood_calibrating = False

                for landmarks in mesh_results.multi_face_landmarks:
                    if show_mesh:
                        mp_drawing.draw_landmarks(
                            image=overlay,
                            landmark_list=landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(
                                color=theme["mesh"], thickness=1, circle_radius=1
                            ),
                        )
                        mp_drawing.draw_landmarks(
                            image=overlay,
                            landmark_list=landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(
                                color=theme["primary"], thickness=1, circle_radius=1
                            ),
                        )

                    if show_mood:
                        mx1, my1, mx2, my2 = landmarks_bounding_box(landmarks, w, h)
                        labels_pos = (mx1, min(h - 45, my2 + 25))
                        if mood_calibrating:
                            draw_hud_text(
                                overlay,
                                f"CALIBRATING... HOLD NEUTRAL FACE ({mood_calib_count}/{MOOD_CALIBRATION_FRAMES})",
                                labels_pos, (255, 255, 255), 0.5, 1,
                            )
                        else:
                            metrics = mouth_metrics(landmarks, w, h)
                            if metrics is not None:
                                mood = classify_mood(metrics, mood_baseline)
                                draw_hud_text(
                                    overlay, f"MOOD: {mood}",
                                    labels_pos, MOOD_COLORS[mood], 0.6, 2,
                                )
                    
            frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

            #TODO: Top HUD bar
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
            draw_hud_text(frame, "AR FACE TRACKER", (12, 26), theme["primary"], 0.65, 2)
            draw_hud_text(frame, f"THEME: {theme['name']}", (w - 220, 26), theme["accent"], 0.5)

            #TODO: Bottom HUD bar
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.rectangle(frame, (0, h - 34), (w, h), (0, 0, 0), -1)
            draw_hud_text(frame, f"FPS: {fps:4.1f}", (12, h - 12), theme["accent"])
            draw_hud_text(frame, f"FACES: {face_count}", (150, h - 12), theme["accent"])
            mood_tag = "ON" if show_mood else "OFF"
            draw_hud_text(
                frame,
                f"[M] mesh [B] box [E] mood:{mood_tag} [R] recalib [C] color [S] shot [F] focus [Q] quit",
                (w - 700, h - 12), (200, 200, 200), 0.45,
            )

            cv2.imshow(WINDOW_NAME, frame)

            if frame_count == 5:
                #TODO: Give the window a few frames to exist before grabbing focus.
                force_window_focus(WINDOW_NAME)

            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("m"):
                show_mesh = not show_mesh
            elif key == ord("b"):
                show_box = not show_box
            elif key == ord("c"):
                theme_idx = (theme_idx + 1) % len(THEMES)
            elif key == ord("s"):
                shot_count += 1
                filename = f"capture_{shot_count:02d}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
            elif key == ord("f"):
                force_window_focus(WINDOW_NAME)
            elif key == ord("e"):
                show_mood = not show_mood
                if show_mood:
                    mood_calibrating = True
                    mood_calib_count = 0
                    mood_calib_sum_width = 0.0
                    mood_calib_sum_lift = 0.0
            elif key == ord("r"):
                if show_mood:
                    mood_calibrating = True
                    mood_calib_count = 0
                    mood_calib_sum_width = 0.0
                    mood_calib_sum_lift = 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()