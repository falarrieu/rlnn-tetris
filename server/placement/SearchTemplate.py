# Do not change the class name or add any other libraries
from queue import PriorityQueue, Queue
import random
from enum import Enum
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from Board import Board
from pieces import PieceProvider
from pieces import PlacementGenerator

import copy

RANDOMIZE_STEPS = 50 # This can be tweaked to increase/decrease the difficulty

# Taken from python docs
@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

class Node(object):
    def __init__(self, board, pieces, trace = []):
        ''' Feel free to add any additional arguments you need'''
        self.board = board
        self.piece_list = pieces
        self.trace : list[tuple[int, int]] = trace
        
    def get_plan(self):
        ''' Return the plan to reach self from the start state'''
        return self.trace
    
    def get_path_cost(self):
        ''' Return the path cost to reach self from the start state'''
        return len(self.trace)

class Problem(object):
    def __init__(self):
        self.piece_provider = PieceProvider()
        self.set_initial_state()
    
    def set_initial_state(self):
        """The initial state for tetris is just the board and random piece.""" 
        board = Board()
        pieces = [self.piece_provider.getNext() for i in range(2)]
        current = Node(board=board, pieces = pieces)
        self.initial_state = current

    def is_goal(self, state):
        """ Checks if this state has been "won".
            In this case, that means every tile is in order from 0..16 (or however large the puzzle is)
        """
        return False # We never hit the goal
        
    def heuristic(self, state, ucs_flag=False):
        if ucs_flag:
            return 0
        else:
            return self.your_heuristic_function(state)
            
    def your_heuristic_function(self, state : Node):
        """Count the number of holes and lines cleared"""
        board = state.board

        holes = board.countHoles()
        lines = board.linesCleared()
                
        return lines - holes

    def get_successors(self, node : Node):
        """"""
        board = node.board
        pieces = node.piece_list

        if len(pieces) == 0:
            return []
        first_piece = pieces[0]

        successors = []
        valid_placements = PlacementGenerator.generateValidPlacements(board, first_piece)

        for valid in valid_placements:
            board_copy = copy.copy(board)
            piece_copy = copy.copy(first_piece)
            piece_copy.setPosition(valid)
            board_copy.placePiece(piece_copy)
            successor = Node(board_copy, pieces[:1])
            successors.append(successor)

        return successors
            

def astar_graph_search(problem: Problem, ucs_flag=False):
    start_state = problem.initial_state
    
    fringe = PriorityQueue()
    # closed = set()
    
    fringe.put(PrioritizedItem(0, start_state)) # Initialize with start state
    
    first_run = True

    best = None
    best_value = 0

    while len(fringe) > 1 and not first_run:
        node = fringe.get().item # Grab next lowest state
        
        # If that wasn't the goal, expand this node and insert it's states into the queue
        successors = problem.get_successors(node)
        
        for option in successors:
            # flattened = tuple(option.board.flatten())

            # if flattened in closed:
            #     continue # We've already seen this board state, just move to next option
                
            # closed.add(flattened)
            
            # f_value = option.get_path_cost() + problem.heuristic(option.board, ucs_flag=ucs_flag)

            heuristic = problem.heuristic(option)

            if len(option.piece_list) == 0: 
                if heuristic > best_value:
                    best_value = heuristic
                    best = option
            else:
                fringe.put(PrioritizedItem(0, option))

            
        first_run = False
    
    return best

    

if __name__ == "__main__":
    ### DO NOT CHANGE THE CODE BELOW ###
    import time
    problem = Problem()
    start = time.time()
    node = astar_graph_search(problem)
    # print("Time taken: ", time.time() - start)
    # print("Plan: ", node.get_plan())
    # print("Path Cost: ", node.get_path_cost())
    # # UCS search
    # start = time.time()
    # node = astar_graph_search(problem, ucs_flag=True)
    # print("Time taken: ", time.time() - start)
    # print("Plan: ", node.get_plan())
    # print("Path Cost: ", node.get_path_cost()),



