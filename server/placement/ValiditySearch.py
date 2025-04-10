# Do not change the class name or add any other libraries
from queue import PriorityQueue, Queue
import random
from enum import Enum
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from Board import Board, createAnimations
from pieces import PieceProvider
from pieces import PlacementGenerator

RANDOMIZE_STEPS = 50 # This can be tweaked to increase/decrease the difficulty

# Taken from python docs
@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

class PlacementMoves(Enum):
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    TURN_LEFT = 4
    TURN_RIGHT = 5

class PositionNode(object):
    def __init__(self, board, target_piece, current_piece, trace = []):
        ''' Feel free to add any additional arguments you need'''
        self.board = board
        self.target_piece = target_piece
        self.current_piece = current_piece
        self.trace : list[PlacementMoves] = trace

    def get_plan(self):
        ''' Return the plan to reach self from the start state'''
        return self.trace

    def get_path_cost(self):
        ''' Return the path cost to reach self from the start state'''
        return len(self.trace)
    
    def get_candidate_legal_moves(self) -> list[PlacementMoves]:            
        last_moves = self.trace[-2:]
        
        possible_moves = [
            PlacementMoves.MOVE_LEFT,
            PlacementMoves.MOVE_RIGHT, 
            PlacementMoves.MOVE_UP, 
            PlacementMoves.TURN_LEFT,
            PlacementMoves.TURN_RIGHT]
        
        if len(self.trace) < 2:
            return possible_moves
        
        if last_moves[0] == PlacementMoves.MOVE_UP:
            return possible_moves

        if last_moves.count(last_moves[0]) != len(last_moves):
            return possible_moves

        return [move for move in possible_moves if last_moves[0] != move]

    def get_successor(self, a: PlacementMoves):
        """Compute the next frame according to the action."""
        checkPiece = self.current_piece.copy()
        if a == PlacementMoves.MOVE_LEFT: # Left
            checkPiece.moveLeft()
            pass
        if a == PlacementMoves.MOVE_RIGHT: # Right
            checkPiece.moveRight()
            pass
        if a == PlacementMoves.TURN_LEFT: # CC
            checkPiece.turnCCW()
            pass
        if a == PlacementMoves.TURN_RIGHT: # CW
            checkPiece.turnCW()
            pass
        if a == PlacementMoves.MOVE_UP: # Up
            checkPiece.moveUp()
            pass

        if self.board.validPlacement(checkPiece):
            return checkPiece
        
        return None


class ValidPlacementProblem(object):
    def __init__(self, board, target_piece, current_piece):
        self.board = board
        self.target_piece = target_piece
        self.current_piece = current_piece
        self.set_initial_state()

    def set_initial_state(self):
        # Create the first node
        self.initial_state = PositionNode(self.board, self.target_piece, self.current_piece)

    def is_goal(self, node):
        """ Check if nodes current piece and valid placement are in the same position and orientation"""
        target_piece = node.target_piece
        current_piece = node.current_piece

        if current_piece.x == target_piece.x:
            if current_piece.y == target_piece.y:
                if current_piece.orientation == target_piece.orientation:
                    return True

        return False

    def heuristic(self, state, ucs_flag=False):
        if ucs_flag:
            return 0
        else:
            return self.your_heuristic_function(state)

    def your_heuristic_function(self, state: PositionNode):
        # We'll calculate the manhattan distance between the two pieces, as well as adding the abs difference in orientation

        rotation_amt = min((state.target_piece.orientation - state.current_piece.orientation) % 4, (state.current_piece.orientation - state.target_piece.orientation) % 4)

        h = abs(state.target_piece.x - state.current_piece.x) + abs(state.target_piece.y - state.current_piece.y) + rotation_amt

        return h
    
    def get_successors(self, node : PositionNode):
        board = node.board

        actions = node.get_candidate_legal_moves()

        successors = []

        for action in actions:

            print(action)

            new_piece = node.get_successor(action)
            
            if new_piece:
                successors.append(PositionNode(board, node.target_piece, new_piece, trace= ( node.trace + [action])))

        return successors


def validity_astar_graph_search(problem: ValidPlacementProblem, ucs_flag=False):
    start_state = problem.initial_state

    fringe = PriorityQueue()
    closed = set()

    fringe.put(PrioritizedItem(0, start_state)) # Initialize with start state

    while not fringe.empty():
        node = fringe.get().item # Grab next lowest state

        # print(f"Dealing with piece x: {node.current_piece.x}, y: {node.current_piece.y}, or: {node.current_piece.orientation}")

        if problem.is_goal(node):
            return node

        # If that wasn't the goal, expand this node and insert it's states into the queue
        successors = problem.get_successors(node)

        print("start looking")
        for option in successors:
            flattened = (option.current_piece.x, option.current_piece.y, option.current_piece.orientation)

            print(f"position of successor :{flattened}")
            # print(f"position of successor :{option.trace}")

            if flattened in closed:
                continue # We've already seen this board state, just move to next option

            closed.add(flattened)

            f_value = option.get_path_cost() + problem.heuristic(option, ucs_flag=ucs_flag)

            fringe.put(PrioritizedItem(f_value, option))

    return None

if __name__ == "__main__":
    ### DO NOT CHANGE THE CODE BELOW ###
    import time

    board = Board()

    piece_provider = PieceProvider()
    piece = piece_provider.getNext()

    target = PlacementGenerator.generateValidPlacements(board, piece)[0]

    target_piece = piece.copy()
    target_piece.setPosition(target)

    problem = ValidPlacementProblem(board, piece, target_piece)
    start = time.time()
    node = validity_astar_graph_search(problem)
    print("Time taken: ", time.time() - start)
    print("Plan: ", node.get_plan())
    print("Path Cost: ", node.get_path_cost())
    # UCS search
    # start = time.time()
    # node = astar_graph_search(problem, ucs_flag=True)
    # print("Time taken: ", time.time() - start)
    # print("Plan: ", node.get_plan())
    # print("Path Cost: ", node.get_path_cost())