
class Vector:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        
    def getLocation(self):
        return [self.x, self.y]
    
    def __eq__(self, value):
        return value.x == self.x and value.y == self.y
    
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
    
    def __repr__(self) -> str:
        return str(self)