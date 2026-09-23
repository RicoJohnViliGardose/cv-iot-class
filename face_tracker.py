import cv2
import mediapipe as mp

CAM_SOURCE = 0

PRIMARY_COLOR = (255, 255, 0)
ACCENT_COLOR = (0, 255, 255)
MESH_COLOR = (255, 255, 0)
BRACKET_LEN = 30
BRACKET_THICKNESS = 3

def draw_glow_line(img, pt1, pt2, color, thickness=1):
    cv2.line(img, pt1, pt2, color, thickness + 3, cv2.LINE_AA)
    cv2.line(img, pt1, pt2, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

def draw_corner_brackets(img, x1, y1, x2, y2, color):
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

def draw_scan_line(img, frame_count, color, top, bottom, left, right):
    span = bottom - top
    if span <= 0:
        return
    offset = frame_count % 60
    y = top + int((offset / 60) * span)
    cv2.line(img, (left, y), (right, y), color, 1, cv2.LINE_AA)



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
        cap = cv2.VideoCapture(CAM_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam. Check CAM_SOURCE / camera permission.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0

    print("AR Face Tracker running. Press 'q' to quit")

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

            det_results = face_detection.process(rgb)
            mesh_results = face_mesh.process(rgb)

            overlay = frame.copy()

            if det_results.detections:
                for det in det_results.detections:
                    box = det.location_data.relative_bounding_box
                    x1 = max(0, int(box.xmin * w) - 15)
                    y1 = max(0, int(box.ymin * h) - 15)
                    x2 = min(w, int((box.xmin + box.width) * w) + 15)
                    y2 = min(h, int((box.ymin + box.height) * h) + 15)
                    draw_corner_brackets(overlay, x1, y1, x2, y2, PRIMARY_COLOR)
                    draw_scan_line(overlay, frame_count, ACCENT_COLOR, y1, y2, x1, x2)
                    conf = det.score[0] if det.score else 0.0
                    draw_hud_text(
                        overlay, f"FACE LOCK {conf * 100:4.1f}%",
                        (x1, max(20, y1 - 12)), PRIMARY_COLOR, 0.5,
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if mesh_results.multi_face_landmarks:
                for landmarks in mesh_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=MESH_COLOR, thickness=1, circle_radius=1)
                    )
                    mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=PRIMARY_COLOR, thickness=1, circle_radius=1
                            ),
                    )
                    
            frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


            cv2.imshow("AR Face Tracker", frame)
            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()