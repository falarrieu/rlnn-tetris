from Board import Board
from pieces import PieceProvider, generateValidPlacementsBFS, L
import numpy as np

class Game:
    """Initialize a game so that the agent has a enviroment to interact with."""
    def __init__(self):
        self.board = Board()
        self.pieceProvider = PieceProvider()
        self.nextPiece()
        
    def reset(self):
        s = self.game.getBoard(), self.game.getPiece().copy(), self.game.getGoalPiece().copy()
        return self.expand_state(s)
    
    def expand_state(self, s):
        '''Every square on the board has a 0 if empty or 1 if filled. So we take each column, make a binary string from
        top to bottom, and convert to decimal to create the inputs to the NN. This is great because no information
        is lost about the column, and you can determine whether a column is higher than another just by comparing
        the value. Hopefully This will help to preserve patterns'''
        board, piece, goal = s
        decimal_columns = [int(''.join([str(int(item)) for item in row]), 2) for row in board.board.T]
        flat_piece_points = np.reshape([[point.x,point.y] for point in piece.getCurrentPoints()], 8)
        flat_goal_points = np.reshape([[point.x,point.y] for point in goal.getCurrentPoints()], 8)
        return np.hstack((decimal_columns, flat_piece_points, flat_goal_points))    
        
    def getGoalPiece(self):
        return self.goalPiece

    def nextPiece(self):
        """Get the next piece for the next trial."""
        # self.currentPiece = self.pieceProvider.getNext()
        self.currentPiece = L(4,0)
        self.goalPiece = self.setGoalPiece()
        self.prevPiece = self.currentPiece.copy()
    
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
                checkPiece = self.currentPiece.copy()
                checkPiece.moveDown(1)
                will_lock = not self.board.validPlacement(checkPiece)
                if not will_lock:
                    self.currentPiece.moveDown(1)

        return self.board, self.currentPiece.copy(), self.goalPiece.copy(), will_lock
    
    def lockPiece(self):
        # for point in self.currentPiece.getCurrentPoints():
        #     self.board.board[point.y, point.x] = 1
        pass
    
    def setGoalPiece(self):
        """Set the piece for which we are aiming to match."""
        # valid_placements = generateValidPlacementsBFS(self.board, self.currentPiece)
        # index = np.random.randint(0,len(valid_placements))
        # chosen_placement = valid_placements[index]
        # pieceType = type(self.currentPiece)
        # piece = pieceType(*chosen_placement)
        piece = L(8, 19)
        piece.orientation = 3 # Bottom right corner, L rotated counter clockwise once
        self.board.setGoalPiece(piece)
        return piece
    
    def getReinforcements(self, prevState, nextState):
        """
        If we are getting closer to the goal X give positive reinforcement and vice versa.
        If we are getting closer to the goal Y give positive reinforcement and vice versa.
        If we are the right orientation give positive reinforcement
        """
        total_reinf = 0
        
        prevBoard, prevPiece, prevGoal = prevState 
        nextBoard, nextPiece, nextGoal = nextState
        
        prevX, prevY, prevO = prevPiece.getPositionAndOrientation()
        nextX, nextY, nextO = nextPiece.getPositionAndOrientation()
        goalX, goalY, goalO = self.goalPiece.getPositionAndOrientation()
        
        # X Distance
        prevXDist = abs(prevX - goalX)
        nextXDist = abs(nextX - goalX) 
             
        if(prevXDist > nextXDist): # Getting Closer
            total_reinf += 1
        else:
            total_reinf -= 1        
        
        # Y Distance
        prevYDist = abs(prevY - goalY)
        nextYDist = abs(nextY - goalY)
        
        if(prevYDist > nextYDist): # Getting Closer
            total_reinf += 1
        else:
            total_reinf -= 1  

        # Orientation
        prevODist = abs(prevO - goalO)
        if prevODist == 3:
            prevODist = 1
        nextODist = abs(nextO - goalO)  
        if nextODist == 3:
            nextODist = 1   
        
        if(prevODist > nextODist):
            total_reinf += 1
        else:
            total_reinf -= 1
            
        # Reaching goal
        success = False
        if(goalX == nextX and goalY == nextY and nextO == goalO):
            total_reinf += 20
            success = True 
        
        return total_reinf, success
    
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
        # Try Drop
        tempPiece = self.currentPiece.copy()
        tempPiece.moveDown(2)
        if self.positionIsValid(tempPiece):
            valid_actions.append(4)
        return valid_actions


    def positionIsValid(self, piece):
        for point in piece.getCurrentPoints():
            if not self.board.inBoard(point.y, point.x) or self.board.filled(point.y, point.x):
                return False
        return True
    