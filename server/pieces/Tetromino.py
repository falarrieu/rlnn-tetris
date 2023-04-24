
class Tetromino:

    def __init__(self, x, y, orientation=0) -> None:
        self.x = x
        self.y = y
        self.orientation = orientation
        
    def getPositionAndOrientation(self):
        return self.x, self.y, self.orientation

    def getCurrentPoints(self):
        return self.getPoints(self.x, self.y, self.orientation)
    
    def getPoints(self, x, y, orientation):
        return []
    
    def __repr__(self) -> str:
        points = self.getPoints(1,1,self.orientation)
        zone = [[' ']*4 for i in range(4)]
        for point in points:
            zone[point.y][point.x] = '*'
        return '\n'.join([''.join(line) for line in zone])
    
    def turnCW(self):
        self.orientation = (self.orientation + 1) % 4

    def turnCCW(self):
        self.orientation = (self.orientation - 1) % 4
        
    def moveLeft(self):
        self.x -= 1
    
    def moveRight(self):
        self.x += 1
    
    def moveDown(self, num):
        self.y += num