import numpy as np

def generateValidPlacements(board, piece):
    validPlacements = []
    for y in range(board.height):
        for x in range(board.width):
            for o in range(4):
                if pieceFits(board, piece, x, y, o):
                    validPlacements.append((x,y,o))
    return validPlacements

def pieceFits(board, piece, x, y, orientation):
    landed = False
    for point in piece.getPoints(x,y,orientation):
        if not board.inBoard(point.y, point.x) or board.filled(point.y, point.x):
            return False
        elif (board.inBoard(point.y+1, point.x) and board.filled(point.y+1, point.x)) or point.y == board.height-1:
            landed = True
    return landed

def generateValidPlacementsBFS(boardObject, piece):
    board = boardObject.copy()
    validPlacements = []
    queue = [(0,0)]
    while len(queue) > 0:
        curLoc = queue.pop(0)
        children = [(curLoc[0],curLoc[1]+1), (curLoc[0]+1,curLoc[1]), (curLoc[0],curLoc[1]-1), (curLoc[0]-1,curLoc[1])]
        for child in children:
            if board.inBoard(child[1],child[0]) and child[1] >= 0 and board.getLocation(child[1], child[0]) == 0:
                board.board[child[1]][child[0]] = -1
                queue.append(child)
        for orientation in range(4):
            if pieceFits(board, piece, curLoc[0], curLoc[1], orientation):
                validPlacements.append((curLoc[0], curLoc[1], orientation))
    return validPlacements
        

                    