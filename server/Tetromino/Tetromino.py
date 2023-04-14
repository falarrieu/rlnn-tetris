
class Tetromino:

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.orientation = 0

    def getPoints(self):
        return []