import numpy as np

def horizon(board):
    top = [checkColumn(board, i,  1) for i in range(board.width)]
    bottom = [checkColumn(board, i,  -1) for i in range(board.width)]
    left = [checkRow(board, i,  1) for i in range(board.height)]
    right = [checkRow(board, i,  -1) for i in range(board.height)]
    return {'top':top,'bottom':bottom,'left':left,'right':right}

def checkColumn(board, column, direction):
    for i in range(board.height):
        if board.filled(i * direction,column):
            return i
    return board.height-1

def checkRow(board, row, direction):
    for i in range(board.width):
        if board.filled(row, i * direction):
            return i
    return board.width-1