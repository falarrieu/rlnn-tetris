# Do not change the class name or add any other libraries
from queue import PriorityQueue, Queue
import random
from enum import Enum
import numpy as np
from dataclasses import dataclass, field
from typing import Any

RANDOMIZE_STEPS = 50 # This can be tweaked to increase/decrease the difficulty

# Taken from python docs
@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

class Node(object):
    def __init__(self, board, trace = []):
        ''' Feel free to add any additional arguments you need'''
        self.board = board
        self.trace : list[tuple[int, int]] = trace
        
    def get_plan(self):
        ''' Return the plan to reach self from the start state'''
        return self.trace
    
    def get_path_cost(self):
        ''' Return the path cost to reach self from the start state'''
        return len(self.trace)

class Problem(object):
    def __init__(self):
        self.set_initial_state()
    
    def set_initial_state(self):
        self.initial_state = np.reshape(np.arange(0, 16), (4, 4))
        
        # Randomize board (by performing random actions so it is solvable)
        # (see. https://math.stackexchange.com/a/754829)
        random.seed(12345)
        
        current = Node(self.initial_state)
        for i in range(RANDOMIZE_STEPS):
            options = self.get_successors(current)
            
            current = random.choice(options)
        
        self.initial_state = current.board
        
        # print(self.initial_state)

    def is_goal(self, state):
        """ Checks if this state has been "won".
            In this case, that means every tile is in order from 0..16 (or however large the puzzle is)
        """
        for i, val in enumerate(state.flatten()):
            if i != val:
                return False
        
        return True
        
    def heuristic(self, state, ucs_flag=False):
        if ucs_flag:
            return 0
        else:
            return self.your_heuristic_function(state)
            
    def your_heuristic_function(self, state):
        # We'll calculate the manhattan distance between each tile's position, and where it should be
        total = 0
        
        for y, row in enumerate(state):
            for x, i in enumerate(row):
                # Calculate where tile *should* be
                
                correct_pos = np.array((int(i / 4), i % 4))
                
                currentPosition = np.array((y, x))
                
                total += np.sum(np.abs(currentPosition - correct_pos))
                
                # print(correct_pos, currentPosition, total)
        
        return total

    def get_successors(self, node : Node):
        # There are 4 actions we can take. Since this is a slide puzzle you could think about
        # it like moving the "empty" piece up, down, left, or right, swapping it with whatever piece was previously
        # in that position. It's important to note that we can never move the empty slot off the board
        
        board = node.board
        
        pos = np.hstack(np.array(np.where(board == 0))) # Get position of empty tile
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # All the different directions
        
        successors = []
        
        for direction in directions:
            position = pos + np.array(direction) 
            
            # Make sure this position is valid
            if not (0 <= position[0] < len(board)):
                continue
            if not (0 <= position[1] < len(board[0])):
                continue
            
            # Create a new board with pieces swapped in that direction
            new_board = np.copy(board)
            new_board[pos[0], pos[1]] = new_board[position[0], position[1]]
            new_board[position[0], position[1]] = 0 # This is where our empty tile is now
        
            successors.append(Node(new_board, node.trace + [direction]))
        
        return successors
            

def astar_graph_search(problem: Problem, ucs_flag=False):
    start_state = Node(problem.initial_state)
    
    fringe = PriorityQueue()
    closed = set()
    
    fringe.put(PrioritizedItem(0, start_state)) # Initialize with start state
    
    while not fringe.empty():
        node = fringe.get().item # Grab next lowest state
        
        if problem.is_goal(node.board):
            return node
        
        # If that wasn't the goal, expand this node and insert it's states into the queue
        successors = problem.get_successors(node)
        
        for option in successors:
            flattened = tuple(option.board.flatten())

            if flattened in closed:
                continue # We've already seen this board state, just move to next option
                
            closed.add(flattened)
            
            f_value = option.get_path_cost() + problem.heuristic(option.board, ucs_flag=ucs_flag)
            
            fringe.put(PrioritizedItem(f_value, option))

if __name__ == "__main__":
    ### DO NOT CHANGE THE CODE BELOW ###
    import time
    problem = Problem()
    start = time.time()
    node = astar_graph_search(problem)
    print("Time taken: ", time.time() - start)
    print("Plan: ", node.get_plan())
    print("Path Cost: ", node.get_path_cost())
    # UCS search
    start = time.time()
    node = astar_graph_search(problem, ucs_flag=True)
    print("Time taken: ", time.time() - start)
    print("Plan: ", node.get_plan())
    print("Path Cost: ", node.get_path_cost())



