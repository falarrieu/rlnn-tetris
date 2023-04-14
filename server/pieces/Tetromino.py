
class Tetromino:

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.orientation = 0

    def getPoints(self):
        return []
    
    def __repr__(self) -> str:
        points = self.getPoints()
        zone = [[' ']*4 for i in range(4)]
        for point in points:
            zone[point.y-self.y+1][point.x-self.x+1] = '*'
        return '\n'.join([''.join(line) for line in zone])
    
    def turnCW(self):
        self.orientation = (self.orientation + 1) % 4

    def turnCCW(self):
        self.orientation = (self.orientation - 1) % 4