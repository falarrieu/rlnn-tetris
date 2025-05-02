from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any
import time
import json
import copy
import random

from Board import Board
from pieces import SevenBagPieceProvider, PieceProvider
from pieces import PlacementGenerator
from ValiditySearch import ValidPlacementProblem, validity_astar_graph_search

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
    def __init__(self, disabled_heuristic=None):
        self.disabled_heuristic = disabled_heuristic
    
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
        board: Board = state.board

        holes = board.countHoles() if self.disabled_heuristic != 'holes' else 0
        lines = (state.lines_cleared / 4) if self.disabled_heuristic != 'lines' else 0
        depth = board.get_fully_empty_depth() if self.disabled_heuristic != 'depth' else 0
        density = board.get_density_under_highest_block() if self.disabled_heuristic != 'density' else 0

        return  lines + (-holes) + depth + density
    
    def weighted_heuristic(self, state: Node, weights):
        board = state.board

        holes = board.countHoles() 
        lines = (state.lines_cleared / 4) 
        depth = board.get_fully_empty_depth()
        density = board.get_density_under_highest_block()

        return (
            weights[0] * lines +
            weights[1] * (-holes) +
            weights[2] * depth +
            weights[3] * density
        )


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
    fringe.put(PrioritizedItem(0, start_state)) 
    
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

# Ablation Study ##################################################

def run_abalation_study():
    games = 10
    play_frames = 5000
    heuristics = [None, 'lines', 'holes', 'depth', 'density']
    results = []

    for excluded in heuristics:
        label = "All heuristics" if excluded is None else f"No {excluded}"

        total_lines = 0
        total_frames = 0

        print(f"\n=== Running: {label} ===")
        for game in range(games):
            piece_provider = PieceProvider()
            pieces = [piece_provider.getNext() for _ in range(2)]
            current_board = Board()
            lines_cleared = 0

            for i in range(play_frames):
                search = Problem(disabled_heuristic=excluded)
                search.set_initial_state(current_board, pieces)

                best_state = piece_search(search)

                if best_state is None:
                    print(f"Game Over at frame {i}")
                    break

                next_action = best_state.get_plan()[0]
                lines_cleared += current_board.placePiece(next_action)
                pieces = pieces[1:] + [piece_provider.getNext()]
                total_frames += i

            print(f"Game {game} finished with {lines_cleared} lines")
            total_lines += lines_cleared
            


        avg_lines = total_lines / games
        avg_frames = total_frames / games
        results.append({"excluded": excluded or "none", "avg_lines_cleared": avg_lines, "avg_frames_survived": avg_frames})

    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Final Results ---")
    for r in results:
        print(f"{r['excluded']:>10}: {r['avg_lines_cleared']} avg lines cleared")

# Genetic Algorithm ##################################################

def weighted_piece_search(problem: Problem, weights):
    start_state = problem.initial_state
    fringe = PriorityQueue()
    fringe.put(PrioritizedItem(0, start_state)) 
    best = None
    best_value = 0

    while not fringe.empty():
        node = fringe.get().item
        successors = problem.get_successors(node)

        for option in successors:
            score = problem.weighted_heuristic(option, weights)
            if len(option.piece_list) == 0:
                if score > best_value or best is None:
                    best_value = score
                    best = option
            else:
                fringe.put(PrioritizedItem(0, option))

    return best

def evaluate_weights(weights, play_frames=5000):
    piece_provider = PieceProvider()
    pieces = [piece_provider.getNext() for _ in range(2)]
    current_board = Board()
    lines_cleared = 0

    for i in range(play_frames):
        search = Problem()
        search.set_initial_state(current_board, pieces)

        best_state = weighted_piece_search(search, weights)
        if best_state is None:
            return lines_cleared, i 

        next_action = best_state.get_plan()[0]
        lines_cleared += current_board.placePiece(next_action)
        pieces = pieces[1:] + [piece_provider.getNext()]

    return lines_cleared, play_frames

def run_genetic_algorithm(generations=10, population_size=10, mutation_rate=0.2):
    population = [random_weights() for _ in range(population_size)]
    best = None
    history = [] 

    for gen in range(generations):
        print(f"\n🌱 Generation {gen + 1}")
        generation_data = {
            "generation": gen + 1,
            "individuals": []
        }

        scored = []

        for i, weights in enumerate(population):
            lines, frames = evaluate_weights(weights)
            fitness = lines + 0.1 * frames

            scored.append((fitness, weights))

            generation_data["individuals"].append({
                "index": i,
                "fitness": fitness,
                "lines_cleared": lines,
                "frames_survived": frames,
                "weights": weights
            })

            print(f"  Individual {i}: score={fitness:.2f}, lines={lines}, frames={frames}, weights={weights}")

        scored.sort(reverse=True)
        population = [w for _, w in scored[:3]]

        while len(population) < population_size:
            parent1, parent2 = random.sample(population[:3], 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            population.append(child)

        best = scored[0]
        generation_data["best"] = {
            "fitness": best[0],
            "weights": best[1]
        }

        history.append(generation_data)

    print(f"\n🏆 Best weights: {best[1]} → score={best[0]:.2f}")

    with open("genetic_results.json", "w") as f:
        json.dump(history, f, indent=2)

    return best[1]

def random_weights():
    return [
        round(random.uniform(0.0, 2.0), 2),   # lines
        round(random.uniform(0.0, 2.0), 2),  # holes (penalty)
        round(random.uniform(0.0, 1.0), 2),   # depth
        round(random.uniform(0.0, 2.0), 2),   # density
    ]

def crossover(w1, w2):
    return [(a + b) / 2 for a, b in zip(w1, w2)]

def mutate(weights):
    return [w + random.uniform(-0.3, 0.3) for w in weights]

# Basic Run ##################################################

def run_games_and_frames(games = 5, play_frames = 500):
    for game in range(games):

        piece_provider = PieceProvider()

        pieces = [piece_provider.getNext() for i in range(2)] # Generate first two pieces

        current_board = Board()

        lines_cleared = 0

        frames = []

        start = time.time()
        for i in range(play_frames):
            search = Problem()

            search.set_initial_state(current_board, pieces)

            best_state = piece_search(search)

            if best_state == None:
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

            # print(f'Game: {game}, Frame: {i}, Time taken: {time.time() - start}, Lines Cleared: {lines_cleared}')

            if i % 10 == 0 or play_frames - 1 == i:
                with open(f'{game}_frames.json', "w") as f:
                    json.dump(frames, f)

        print(f'Game: {game},  Time taken: {time.time() - start}, Lines Cleared: {lines_cleared}')


if __name__ == "__main__":  
    run_games_and_frames() 
    # run_abalation_study() 
    # run_genetic_algorithm()