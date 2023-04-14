from pieces import Tetromino
from Vector import Vector

class O(Tetromino):
    def __init__(self, x, y) -> None:
        super().__init__(x, y)
    
    def getPoints(self):
        x = self.x
        y = self.y
        # returns array of arrays of points. Indexed by orientation
        points = [
            [Vector(x-1, y), Vector(x, y), Vector(x, y-1), Vector(x-1, y-1)],
            [Vector(x-1, y), Vector(x, y), Vector(x, y-1), Vector(x-1, y-1)],
            [Vector(x-1, y), Vector(x, y), Vector(x, y-1), Vector(x-1, y-1)],
            [Vector(x-1, y), Vector(x, y), Vector(x, y-1), Vector(x-1, y-1)]
        ]
        return points[self.orientation]