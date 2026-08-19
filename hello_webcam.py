import cv2

CAM_SOURCE = 0

if isinstance(CAM_SOURCE, int):
    cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(CAM_SOURCE)

if not cap.isOpened():
    raise RuntimeError("Could not open camera. check CAMSOURCE / camera permissions")
ok, frame = cap.read()
if ok:
    cv2.imshow("hello, camera", frame)
    cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()