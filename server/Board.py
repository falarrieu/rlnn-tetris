import numpy as np
import random as rand
import matplotlib.pyplot as plt 

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
        self.board = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
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


    def bruteRandomFrame(self, verbose=False):
        board = self.board.copy()
        rand.seed(12)
        horizontal_line = rand.randint(0, self.height - 1)
        if verbose: print(f'Horizontal Line: {horizontal_line}')
        rand_row_index = [rand.randint(0, horizontal_line) for i in range(self.width)]
        if verbose: print(f'Random Top Block Indexes: {rand_row_index}')
        for col in range(self.width):
            top_block = rand_row_index[col]
            board[-top_block][col] = 1
            for block in range(top_block, self.height):
                board[block][col] = 1
        return board

    def sinRandomFrame(self, verbose=False):
        # rand.seed(12)
        board = self.board.copy()
        x = np.arange(0, self.width, 1)
        inverse = rand.randint(0, 1)
        axis = rand.randint(self.height - 2, self.height - 1)
        if inverse:
            amplitude = -np.sin(x) + np.random.normal(scale=.4, size=len(x)) + axis
        else:
            amplitude = np.sin(x) + np.random.normal(scale=.4, size=len(x)) + axis

        processed_amp = np.round(amplitude)
        # processed_amp[processed_amp <= 0] = 0
        processed_amp[processed_amp > self.height - 1] = 0
        if verbose:
            plt.plot(x, processed_amp)
            plt.show()

        for col in range(self.width):
            board[int(processed_amp[col])][col] = 1

        return board
    