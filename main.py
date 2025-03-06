import cv2

from camera import CameraManager

cam = CameraManager()
frame = cam.capture()

if frame is not None:
    cv2.imshow("Camera", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()