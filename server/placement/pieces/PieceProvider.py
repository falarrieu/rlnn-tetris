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
        pieces = [I(4, 1), O(4,1), T(4,1), S(4,1), Z(4,1), L(4,1), J(4,1)]
        for i in range(7):
            index = np.random.randint(0,len(pieces))
            self.pieceQueue.append(pieces.pop(index))
