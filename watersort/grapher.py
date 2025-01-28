import networkx as nx
import numpy as np
from manim import *

from watersort.helpers import WaterPuzzleSolver, HashablePuzzleState


import typing
if typing.TYPE_CHECKING:
    from watersort.main import WaterPuzzle


class Grapher:
    def __init__(self, first_node_hashable: HashablePuzzleState):
        self.current_nodes = set()
        self.current_edges = set()
        self.first_node = first_node_hashable
        self.hash_to_puzzle = {}

        self.current_positions = {}
        self.hashable_to_y_distance = {}

        self.node_mgroups = []
        self.edge_mgroups = []

    def add_node_to_graph(self, node: HashablePuzzleState, y_distance: int):
        self.current_nodes.add(node)
        self.hashable_to_y_distance[node] = y_distance

    def add_edge_to_graph(self, from_node: HashablePuzzleState, to_node: HashablePuzzleState):
        self.current_edges.add((from_node, to_node))

    def run_spring_layout_re_balance(self):
        graph = nx.Graph()
        for node in self.current_nodes:
            graph.add_node(node)
        for edge in self.current_edges:
            graph.add_edge(*edge)

        positions = self.current_positions
        positions[self.first_node] = np.array([0, 0])

        # y_distance = ([0.75, 0.55, 0.45, 0.3, 0.25] + [0.2] * 5 + [0.1] * 15)[i]
        # y_distance = ([0.75, 0.55, 0.45, 0.3, 0.25] + [0.2 - (i / 20) * 0.1 for i in range(20)])[i]
        y_distance = 0.3

        for j in range(10):
            positions = nx.spring_layout(
                graph,
                pos=positions,  # TODO: Set the new ones close to the parents by default, not randomly
                # fixed=first_node  # First node always at the center
            )
            for hash, position in positions.items():
                node_distance = self.hashable_to_y_distance[hash]
                position[1] = -node_distance * y_distance  # + i * y_distance / 2

        print("run_spring_layout_re_balance", positions)
        self.current_positions = positions

    def construct_mobject_graph(self, solver: WaterPuzzleSolver):
        puzzles_to_add = []
        # line_edges_to_add = []

        for node_i, node in enumerate(self.current_nodes):
            # self.current_nodes.add(hashable)
            # self.current_edges.update([
            #     edge
            #     for edge in solver.edges
            #     if edge[1] == hashable
            # ])

            hashable_original_pipes = solver.hashable_to_original_unsorted[node].pipes
            from watersort.main import WaterPuzzle
            puzzle = WaterPuzzle.new_from_hashable_state(hashable_original_pipes)

            all_flasks = puzzle.all_flasks
            for flask in all_flasks:
                flask.rotating = True

            desired_z_index = 100 + 10 * node_i
            flasks = VGroup(all_flasks).set_z_index(desired_z_index + 20)

            rect = SurroundingRectangle(
                flasks,
                buff=1,
                corner_radius=1,
                color=GREEN if puzzle.puzzle.is_solved() else WHITE,
            ).set_z_index(desired_z_index + 10).set_fill(
                color=GREEN if puzzle.puzzle.is_solved() else BLACK,
                opacity=1.0
            )
            both = VGroup(flasks, rect)
            puzzles_to_add.append(both)

            self.hash_to_puzzle[node] = both

            # for edge in self.current_edges:
            #     # TODO: Should add direction to edges? Some arrows perhaps?
            #     if edge[0] == node and edge[1] in hash_to_puzzle:
            #         # TODO: Does this fix my issues? Does it add duplicate edges?
            #         # TODO: HMM maybe it is bi directional if both are added?
            #         line_edges_to_add.append(edge)
            #     if edge[1] == node and edge[0] in hash_to_puzzle:
            #         line_edges_to_add.append(edge)

        desired_scale = 0.2

        # Move puzzles to right positions
        for hash, puzzle in self.hash_to_puzzle.items():
            real_pos = np.array([*self.current_positions[hash] * 5, 0])
            puzzle.move_to(real_pos).scale(desired_scale)

        lines_to_add = []
        for node1, node2 in self.current_edges:
            real_pos1 = np.array([*self.current_positions[node1] * 5, 0])
            real_pos2 = np.array([*self.current_positions[node2] * 5, 0])
            new_line = Line(
                real_pos1,
                real_pos2,
                color=WHITE,
            )
            # new_line.from_puzzle_hash = edge_to_add[0]
            # new_line.to_puzzle_hash = edge_to_add[1]
            lines_to_add.append(new_line)


        # initial_positions = {}
        # for node in self.current_nodes:
        #     if node not in initial_positions:
        #         start_nodes_for_this_node = [
        #             e[0]
        #             for e in self.current_edges
        #             if e[1] == node and e[0] in initial_positions
        #         ]
        #         if not start_nodes_for_this_node:
        #             initial_positions[node] = np.array([0, 0])
        #         else:
        #             initial_positions[node] = (
        #                     initial_positions[start_nodes_for_this_node[0]]
        #                     + np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * 0.01
        #             )
        # desired_scale = 0.2
        # hash_to_new_pos = {}

        # anims = []
        # for hashed, position in self.current_positions.items():
        #     if hashed in hash_to_puzzle:
        #         to_be_moved = hash_to_puzzle[hashed]
        #         new_pos = np.array([*position * 5, 0])
        #         hash_to_new_pos[hashed] = new_pos
        #         if to_be_moved in puzzles_to_add:
        #             hash_to_puzzle[hashed].move_to(new_pos).scale(desired_scale)
        #         else:
        #             to_be_moved.move_to(new_pos).scale(desired_scale)
        #             # anims.append(
        #             #     to_be_moved.animate
        #             #     .move_to(new_pos)
        #             #     .scale(desired_scale / desired_scale_old)
        #             # )

        # lines_to_add = []
        # for edge_to_add in line_edges_to_add:
        #     new_line = Line(
        #         hash_to_new_pos[edge_to_add[0]],
        #         hash_to_new_pos[edge_to_add[1]],
        #         color=WHITE,
        #     )
        #     new_line.from_puzzle_hash = edge_to_add[0]
        #     new_line.to_puzzle_hash = edge_to_add[1]
        #     lines_to_add.append(new_line)

        my_vgroup = VGroup(*puzzles_to_add, *lines_to_add)

        my_vgroup.puzzles = puzzles_to_add
        my_vgroup.lines = lines_to_add

        self.start_puzzle = self.hash_to_puzzle[self.first_node]
        self.goal_puzzle = self.hash_to_puzzle[solver.winning_node()]

        return my_vgroup

    def transition_nodes_to_balls(self):
        anims = []

        for puzzle in self.hash_to_puzzle.values():
            circle = (
                Circle(color=WHITE, radius=0.5)
                .set_fill(color=BLACK, opacity=1)
                .move_to(puzzle.get_center())
                .set_z_index(10000)
            )
            if puzzle == self.start_puzzle:
                circle.set_fill(color=RED, opacity=1)
            if puzzle == self.goal_puzzle:
                circle.set_fill(color=GREEN, opacity=1)

            # anims.append(Transform(puzzle, circle))
            anims.append(FadeIn(circle))
            anims.append(FadeOut(puzzle))

        return anims
