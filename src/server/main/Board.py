import numpy as np
import random as rand
import matplotlib.pyplot as plot


class Board:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = self.createBoard()

    def createBoard(self):
        return np.zeros((self.height, self.width))

    def printBoard(self):
        print(np.matrix(self.board))

    def bruteRandomFrame(self, verbose=False):
        board = self.createBoard()
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
        board = self.createBoard()
        rand.seed(12)
        x = np.arange(0, self.width, 1);
        inverse = rand.randint(0, 1)
        axis = horizontal_line = rand.randint(0, self.height - 1)
        if inverse:
            amplitude = -np.sin(x) + np.random.normal(scale=1, size=len(x)) + axis
        else:
            amplitude = np.sin(x) + np.random.normal(scale=1, size=len(x)) + axis

        processed_amp = np.round(amplitude)
        processed_amp[processed_amp <= 0] = 0
        processed_amp[processed_amp > self.height - 1] = self.height - 1
        if verbose:
            plot.plot(x, processed_amp)
            plot.show()

        for col in range(self.width):
            board[int(processed_amp[col])][col] = 1

        return board
