from Board import Board
from horizon import horizon
from pieces import *
import time
import matplotlib.pyplot as plt

def main():
    board = Board()
    board.testBoard()
    # board.printBoard()

    provider = PieceProvider()
    # for i in range(14):
    #     print(provider.getNext())
    #     time.sleep(1)
    piece = provider.getNext()
    placements = generateValidPlacementsBFS(board, piece)
    for (x,y,o) in placements:
        piece.x = x
        piece.y = y
        piece.orientation = o
        board.showBoard(piece)


if __name__ == "__main__":
    main()