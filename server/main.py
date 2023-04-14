from Board import Board
from horizon import horizon

def main():
    board = Board()
    board.testBoard()
    board.printBoard()

    print(horizon(board))


if __name__ == "__main__":
    main()