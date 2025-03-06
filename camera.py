from time import sleep

import cv2


class CameraManager:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        sleep(0.5)  # delay for camera warm up
        
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
    
    def capture(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
        
        ret, frame = self.cap.read()

        if not ret:
            print("Error: Could not capture frame.")
            return None
        
        return frame
    
    def capture_to_file(self, file_name):
        frame = self.capture()
        if frame is not None:
            cv2.imwrite(file_name, frame)
            print(f"Image saved to {file_name}")