from Board import Board
from horizon import horizon
from pieces import *
import time

def main():
    board = Board()
    board.testBoard()
    # board.printBoard()

    provider = PieceProvider()
    for i in range(14):
        print(provider.getNext())
        time.sleep(1)



if __name__ == "__main__":
    main()