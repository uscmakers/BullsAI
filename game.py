from enum import Enum

import cv2

from aruco import ArucoManager
from camera import CameraManager
from led_detection import LEDDetection
from player import Player


class GameState(Enum):
    USER = 1
    ROBOT = 2

class Game:
    def __init__(self):
        self.user = Player()
        self.robot = Player()
        self.state = GameState.USER
        self.turn_num = 1
        self.camera = CameraManager()

    def run(self, turns=10):
        while self.turn_num <= turns:
            
            if self.state == GameState.USER:
                self.turn(self.user)
                self.state = GameState.ROBOT
            else:
                self.turn(self.robot)
                self.state = GameState.USER

            self.turn_num += 1
            break

    def turn(self, player: Player):
        # frame = self.camera.capture()
        frame = cv2.imread("test.jpg")

        output = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame is not None:
            aruco = ArucoManager(output_image=output)
            led_detection = LEDDetection(output_image=output)
            
            aruco_reference_points = aruco.detect_markers(gray)
            led_detection.detect_led(gray, aruco_reference_points)


            points = None # TODO: implement CV logic and point calculation
            player.add_points(points)


if __name__ == "__main__":
    game = Game()
    game.run()