# Do not change the class name or add any other libraries
from queue import PriorityQueue, Queue
import random
from enum import Enum
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import time
import json

from Board import Board, createAnimations
from pieces import PieceProvider
from pieces import PlacementGenerator

from ValiditySearch import ValidPlacementProblem, validity_astar_graph_search

import cProfile
import pstats

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
        board: Board = state.board

        holes = board.countHoles()
        lines = state.lines_cleared
        depth = board.get_fully_empty_depth()
                
        return lines - holes + depth

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

            copied_piece = first_piece.copy()
            copied_piece.setPosition(valid)
            valid_placement_problem = ValidPlacementProblem(board, first_piece, copied_piece)
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

    while not fringe.empty():
        node = fringe.get().item # Grab next lowest state

        successors = problem.get_successors(node)

        for option in successors:

            heuristic = problem.heuristic(option)
            if len(option.piece_list) == 0: 
                if heuristic > best_value or best == None:
                    best_value = heuristic
                    best = option
            else:
                fringe.put(PrioritizedItem(0, option))
    
    return best

    

if __name__ == "__main__":    
    # profiler = cProfile.Profile()
    # profiler.enable()

    games = 10

    for game in range(games):

        piece_provider = PieceProvider()

        pieces = [piece_provider.getNext() for i in range(2)] # Generate first two pieces

        current_board = Board()

        lines_cleared = 0

        frames = []

        start = time.time()
        for i in range(15000):
            search = Problem()

            search.set_initial_state(current_board, pieces)

            best_state = piece_search(search)

            if not best_state:
                print("Game Over")
                with open(f'{game}_frames.json', "w") as f:
                    json.dump(frames, f)
                break 
            
            next_action = best_state.get_plan()[0]

            lines_cleared += current_board.placePiece(next_action)

            pieces = pieces[1:] + [piece_provider.getNext()]

            frames.append({
                "board": current_board.to_dict(),
                "lines_cleared": lines_cleared
            })

            print(f'Frame: {i}, Time taken: {time.time() - start}, Lines Cleared: {lines_cleared}')

            with open(f'{game}_frames.json', "w") as f:
                json.dump(frames, f)

    # Uncomment Below to Create Animation from File
    # with open("frames.json", "r") as f:
    #     frame_data = json.load(f)

    # # Convert dictionaries back into (Board, lines_cleared) tuples
    # frames = [
    #     (Board.from_dict(frame["board"]), frame["lines_cleared"])
    #     for frame in frame_data
    # ]

    # # Animate them
    # createAnimations(frames)

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative').print_stats(10)
