from Board import Board
from pieces import PieceProvider, generateValidPlacementsBFS
import numpy as np

class Game:
    """Initialize a game so that the agent has a enviroment to interact with."""
    def __init__(self):
        self.board = Board()
        self.pieceProvider = PieceProvider()
        self.gamePlaying = False
        self.nextPiece()
        pass

    def nextPiece(self):
        """Get the next piece for the next trial."""
        self.currentPiece = self.pieceProvider.getNext()
        self.goalPiece = self.setGoalPiece()
    
    def getBoard(self):
        return self.board
    
    def getPiece(self):
        return self.currentPiece
    
    def getNextFrame(self, s, a):
        """Compute the next frame according to the action."""
        if a == 0: # Left
            self.currentPiece.moveLeft()
            pass
        if a == 1: # Right
            self.currentPiece.moveRight()
            pass
        if a == 2: # CC
            self.currentPiece.turnCCW()
            pass
        if a == 3: # CW
            self.currentPiece.turnCW()
            pass
        if a == 4: # Drop
            self.currentPiece.moveDown(2)
            pass
        else:
            self.currentPiece.moveDown(1)
        
        return self.board, self.currentPiece
    
    def setGoalPiece(self):
        """Set the piece for which we are aiming to match."""
        valid_placements = generateValidPlacementsBFS(self.board, self.currentPiece)
        index = np.random.randint(0,len(valid_placements))
        chosen_placement = valid_placements[index]
        pieceType = type(self.currentPiece)
        piece = pieceType(*chosen_placement)
        self.board.showBoard(piece)
        return piece
    
    def getReinforcements(self, prevState, nextState):
        """
        If we are getting closer to the goal X give positive reinforcement and vice versa.
        If we are getting closer to the goal Y give positive reinforcement and vice versa.
        If we are the right orientation give positive reinforcement
        """
        total_reinf = 0
        
        prevBoard, prevPiece = prevState 
        nextBoard, nextPiece = nextState
        
        prevX, prevY, prevO = prevPiece.getPostionAndOrientation()
        nextX, nextY, nextO = nextPiece.getPostionAndOrientation()
        goalX, goalY, goalO = self.goalPiece.getPositionAndOrientation()
        
        prevXDist = abs(prevX - goalX)
        nextXDist = abs(nextX - goalX) 
        
        if(prevXDist > nextXDist): # Getting Closer
            total_reinf += 1
        else:
            total_reinf -= 1        
        
        prevYDist = abs(prevY - goalY)
        nextYDist = abs(nextY - goalY)
        
        if(prevYDist > nextYDist): # Getting Closer
            total_reinf += 1
        else:
            total_reinf -= 1       
        
        if(nextO == goalO):
            total_reinf += 1
            
        if(goalX == nextX and goalY == nextY and nextO == goalO):
            total_reinf += 20
        
        return total_reinf
    