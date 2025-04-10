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

from ValiditySearch import ValidPlacementProblem, validity_astar_graph_search

import copy

# Taken from python docs
@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

class Node(object):
    def __init__(self, board, pieces, trace = [], lines_cleared = 0):
        ''' Feel free to add any additional arguments you need'''
        self.board = board
        self.piece_list = pieces
        self.lines_cleared = lines_cleared
        self.trace : list[tuple[int, int]] = trace
        
    def get_plan(self):
        ''' Return the plan to reach self from the start state'''
        return self.trace
    
    def get_path_cost(self):
        ''' Return the path cost to reach self from the start state'''
        return len(self.trace)

class Problem(object):
    def __init__(self):
        # self.set_initial_state()
        pass
    
    def set_initial_state(self, curBoard: Board, nextPieces):
        """The initial state for tetris is just the board and random piece.""" 
        board = curBoard
        pieces = nextPieces
        current = Node(board=board, pieces = pieces)
        self.initial_state = current
        
    def heuristic(self, state, ucs_flag=False):
        if ucs_flag:
            return 0
        else:
            return self.your_heuristic_function(state)
            
    def your_heuristic_function(self, state : Node):
        """Count the number of holes and lines cleared"""
        board = state.board

        holes = board.countHoles()
        # lines = board.linesCleared()
        lines = state.lines_cleared
                
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
            board_copy = board.copy()
            piece_copy = copy.deepcopy(first_piece)
            piece_copy.setPosition(valid)
            
            valid_placement_problem = ValidPlacementProblem(board, first_piece, first_piece.copy().setPosition(valid))
            valid_node = validity_astar_graph_search(valid_placement_problem)

            if not valid_node:
                continue

            lines_cleared = board_copy.placePiece(piece_copy)

            successor = Node(board_copy, pieces[1:], node.trace + [piece_copy], lines_cleared + node.lines_cleared)
            successors.append(successor)

        return successors
            

def piece_search(problem: Problem, ucs_flag=False):
    start_state = problem.initial_state
    
    fringe = PriorityQueue()
    fringe.put(PrioritizedItem(0, start_state)) # Initialize with start state
    
    best = None
    best_value = 0

    counter = 0

    while not fringe.empty():
        node = fringe.get().item # Grab next lowest state

    #     if counter % 100 == 0:
    #         print(f'iteration {counter}, fringe size: {fringe.qsize()}, best_value: {best_value}')
    #     counter += 1
        
    #     # If that wasn't the goal, expand this node and insert it's states into the queue
        successors = problem.get_successors(node)
        
        for option in successors:

            heuristic = problem.heuristic(option)
            if len(option.piece_list) == 0: 
                if heuristic > best_value or best == None:
                    best_value = heuristic
                    best = option
            else:
                fringe.put(PrioritizedItem(0, option))
    
    print(f'Final for step: iteration {counter}, fringe size: {fringe.qsize()}, best_value: {best_value}')

    return best

    

if __name__ == "__main__":
    # ### DO NOT CHANGE THE CODE BELOW ###
    # import time
    # problem = Problem()
    # start = time.time()
    # node = astar_graph_search(problem)
    # # print("Time taken: ", time.time() - start)
    # # print("Plan: ", node.get_plan())
    # # print("Path Cost: ", node.get_path_cost())
    # # # UCS search
    # # start = time.time()
    # # node = astar_graph_search(problem, ucs_flag=True)
    # # print("Time taken: ", time.time() - start)
    # # print("Plan: ", node.get_plan())
    # # print("Path Cost: ", node.get_path_cost()),


    piece_provider = PieceProvider()


    pieces = [piece_provider.getNext() for i in range(2)] # Generate first two pieces

    current_board = Board()

    lines_cleared = 0

    frames = []

    for i in range(50):
        search = Problem()

        search.set_initial_state(current_board, pieces)

        best_state = astar_graph_search(search)

        next_action = best_state.get_plan()[0]

        lines_cleared += current_board.placePiece(next_action)

        # current_board.showBoard()

        pieces = pieces[1:] + [piece_provider.getNext()]

        frames.append((current_board.copy(), lines_cleared))


    createAnimations(frames)
