from enum import Enum

from camera import CameraManager
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

    def turn(self, player: Player):
        frame = self.camera.capture()

        if frame is not None:
            points = None # TODO: implement CV logic and point calculation
            player.add_points(points)


if __name__ == "__main__":
    game = Game()
    game.run()