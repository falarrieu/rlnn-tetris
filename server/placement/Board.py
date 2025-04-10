import numpy as np
import random as rand
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

class Board:

    def __init__(self):
        self.width = 10
        self.height = 20
        self.board = np.zeros((self.height, self.width))
        pass
        
    def copy(self):
        boardCopy = Board()
        boardCopy.board = np.copy(self.board)
        return boardCopy
    
    def printBoard(self):
        print(self.board.shape)
        print(self.board)
        pass

    def showBoard(self, piece=None):
        gray_map=plt.cm.get_cmap('gray')
        if piece is not None:
            board = self.board * 2 
            for point in piece.getCurrentPoints():
                board[point.y, point.x] = 1
            plt.imshow(board, cmap=gray_map.reversed(), vmin=0, vmax=2)
            plt.show()
        else:
            plt.imshow(self.board, cmap=gray_map.reversed(), vmin=0, vmax=1)
            plt.show()
    
    def filled(self, row, col):
        return self.board[row][col] == 1
    
    def inBoard(self, row, col):
        return row < self.height and col >= 0 and col < self.width 
    
    def getLocation(self, row, col):
        return self.board[row][col]
    
    def validPlacement(self, piece):
        for point in piece.getCurrentPoints():
            if not self.inBoard(point.y, point.x) or self.filled(point.y, point.x):
                return False
        return True
       
    def placePiece(self, piece):
        points = piece.getCurrentPoints()
        for point in points:
            y, x = point.getLocation()
            self.board[x][y] = 1

        return self.linesCleared()

    def countHoles(self):
        holes = 0
        for col in range(self.width):
            block_found = False
            for row in range(self.height):
                if self.board[row][col] == 1:
                    block_found = True
                elif self.board[row][col] == 0 and block_found:
                    holes += 1
        return holes
    
    def linesCleared(self):
        new_board = []
        lines_cleared = 0

        for row in self.board:
            if np.all(row == 1):
                lines_cleared += 1  # full line found
            else:
                new_board.append(row)  # keep incomplete lines

        # Add empty rows at the top
        while len(new_board) < self.height:
            new_board.insert(0, np.zeros(self.width))

        self.board = np.array(new_board)
        return lines_cleared

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
    
    def to_dict(self):
        return {
            "board": self.board.tolist()
        }

    @staticmethod
    def from_dict(data):
        board = Board()
        board.board = np.array(data["board"])
        return board


def createAnimations(frames):
    fig, ax = plt.subplots()
    gray_map = plt.cm.get_cmap('gray')

    # Initialize with the first frame
    im = ax.imshow(frames[0][0].board, cmap=gray_map.reversed(), vmin=0, vmax=1)

    def frameUpdate(i):
        board = frames[i][0].board
        im.set_data(board)
        ax.set_title(f'Frame {i}')
        ax.set_ylabel(f'Lines Cleared: {frames[i][1]}', rotation=0)


    ani = FuncAnimation(fig, frameUpdate, frames=len(frames), interval=200)
    ani.save('tetris_animation.gif', dpi=100, writer='pillow')
    plt.close()

    
