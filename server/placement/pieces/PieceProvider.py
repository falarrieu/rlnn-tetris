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
        pieces = [I(4, 2), O(4,2), T(4,2), S(4,2), Z(4,2), L(4,2), J(4,2)]
        for i in range(7):
            index = np.random.randint(0,len(pieces))
            self.pieceQueue.append(pieces.pop(index))
