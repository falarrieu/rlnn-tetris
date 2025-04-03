from pieces import Tetromino
from server.placement.pieces.Vector import Vector

class J(Tetromino):
    def __init__(self, x, y, orientation=0) -> None:
        super().__init__(x, y, orientation)
    
    def getPoints(self,x,y,orientation):
        # returns array of arrays of points. Indexed by orientation
        points = [
            [Vector(x, y-1), Vector(x, y), Vector(x, y+1), Vector(x-1, y+1)],
            [Vector(x+1, y), Vector(x, y), Vector(x-1, y), Vector(x-1, y-1)],
            [Vector(x, y+1), Vector(x, y), Vector(x, y-1), Vector(x+1, y-1)],
            [Vector(x-1, y), Vector(x, y), Vector(x+1, y), Vector(x+1, y+1)]
        ]
        return points[orientation]