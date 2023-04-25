from Board import Board
from pieces import PieceProvider, generateValidPlacementsBFS
import numpy as np

class Game:
    """Initialize a game so that the agent has a enviroment to interact with."""
    def __init__(self):
        self.board = Board()
        self.pieceProvider = PieceProvider()
        self.nextPiece()

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

        checkPiece = self.currentPiece.copy()
        checkPiece.moveDown(1)
        will_lock = not self.board.validPlacement(checkPiece)
        if not will_lock:
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

        return self.board, self.currentPiece, will_lock
    
    def lockPiece(self):
        for point in self.currentPiece.getCurrentPoints():
            self.board.board[point.y, point.x] = 1
    
    def setGoalPiece(self):
        """Set the piece for which we are aiming to match."""
        valid_placements = generateValidPlacementsBFS(self.board, self.currentPiece)
        index = np.random.randint(0,len(valid_placements))
        chosen_placement = valid_placements[index]
        pieceType = type(self.currentPiece)
        piece = pieceType(*chosen_placement)
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
        
        prevX, prevY, prevO = prevPiece.getPositionAndOrientation()
        nextX, nextY, nextO = nextPiece.getPositionAndOrientation()
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
    
    def getValidActions(self):
        valid_actions = []
        # Try left
        tempPiece = self.currentPiece.copy()
        tempPiece.moveLeft()
        if self.positionIsValid(tempPiece):
            valid_actions.append(0)
        # Try right
        tempPiece = self.currentPiece.copy()
        tempPiece.moveRight()
        if self.positionIsValid(tempPiece):
            valid_actions.append(1)
        # Try CC
        tempPiece = self.currentPiece.copy()
        tempPiece.turnCCW()
        if self.positionIsValid(tempPiece):
            valid_actions.append(2)
        # Try CW
        tempPiece = self.currentPiece.copy()
        tempPiece.turnCW()
        if self.positionIsValid(tempPiece):
            valid_actions.append(3)
        valid_actions.append(4)
        return valid_actions


    def positionIsValid(self, piece):
        for point in piece.getCurrentPoints():
            if not self.board.inBoard(point.y, point.x) or self.board.filled(point.y, point.x):
                return False
        return True
    