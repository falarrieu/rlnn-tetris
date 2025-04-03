import numpy as np
from pieces import *

class PieceProvider:
    def __init__(self) -> None:
        self.pieceQueue = []

    def getNext(self) -> Tetromino:
        if len(self.pieceQueue) == 0:
            self.populatePieceQueue()
        return self.pieceQueue.pop(0)
    
    def peekNext(self) -> Tetromino:
        if len(self.pieceQueue) == 0:
            self.populatePieceQueue()
        return self.pieceQueue[0]

    def populatePieceQueue(self):
        pieces = [I(4, 0), O(4,0), T(4,0), S(4,0), Z(4,0), L(4,0), J(4,0)]
        for i in range(7):
            index = np.random.randint(0,len(pieces))
            self.pieceQueue.append(pieces.pop(index))
