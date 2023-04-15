

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
    board.printBoard()
    for point in piece.getPoints(x,y,orientation):
        if not board.inBoard(point.y, point.x) or board.filled(point.y, point.x):
            return False
        elif (board.inBoard(point.y+1, point.x) and board.filled(point.y+1, point.x)) or point.y == board.height-1:
            landed = True
    return landed

                    