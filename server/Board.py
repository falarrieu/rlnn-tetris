import numpy as np

class Board:

    def __init__(self):
        self.width = 10
        self.height = 20
        self.board = np.zeros((self.height, self.width))
        pass
    
    
    def printBoard(self):
        print(self.board.shape)
        print(self.board)
        pass
    
    def filled(self, row, col):
        return self.board[row][col] == 1
    

    def testBoard(self):   
        self.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0,],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0,],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0,],
            [0, 1, 0, 1, 0, 1, 1, 0, 0, 0,],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0,],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,]])
        pass