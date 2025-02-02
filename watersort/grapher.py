import networkx as nx
import numpy as np
from manim import *

from watersort.helpers import WaterPuzzleSolver, HashablePuzzleState


import typing
if typing.TYPE_CHECKING:
    from watersort.main import WaterPuzzle


class Grapher:
    def __init__(self, first_node_hashable: HashablePuzzleState, winning_node_hashable: HashablePuzzleState):
        self.current_nodes = set()
        self.current_edges = set()
        self.first_node = first_node_hashable
        self.goal_node = winning_node_hashable

        self.current_positions = {}
        self.hashable_to_y_distance = {}

        self.hash_to_node_mgroup = {}
        self.node_mgroups = []
        self.edge_mgroups = []

    def add_node_to_graph(
        self,
        solver: WaterPuzzleSolver | None,
        node: HashablePuzzleState,
        y_distance: int,
        use_circle: bool = False,
        start_at: HashablePuzzleState | None = None,
        outline_color=WHITE,
    ):
        self.current_nodes.add(node)
        self.hashable_to_y_distance[node] = y_distance

        # Add the mgroup
        if use_circle:
            circle = (
                Circle(color=outline_color, radius=0.5)
                .set_fill(color=BLACK, opacity=1)
                # .move_to(puzzle.get_center())
                .set_z_index(10000)
            )
            circle.node = node
            both = circle
        else:
            hashable_original_pipes = solver.hashable_to_original_unsorted[node].pipes if solver else node
            from watersort.main import WaterPuzzle
            puzzle = WaterPuzzle.new_from_hashable_state(hashable_original_pipes)

            all_flasks = puzzle.all_flasks
            for flask in all_flasks:
                flask.rotating = True

            desired_z_index = 100 + 10 * len(self.current_nodes)
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
            both.node = node
            both.scale(0.2)

        if start_at:
            both.move_to(
                np.array([*self.current_positions[start_at] * 5, 0])
            )

        self.node_mgroups.append(both)
        self.hash_to_node_mgroup[node] = both

    def add_edge_to_graph(self, from_node: HashablePuzzleState, to_node: HashablePuzzleState, color=WHITE):
        self.current_edges.add((from_node, to_node))

        start_pos = (
            ORIGIN
            if from_node not in self.current_positions else
            np.array([*self.current_positions[from_node] * 5, 0])
        )
        end_pos = (  # Hack: set both to the same (when spanning new nodes looks nice.)
            ORIGIN
            if from_node not in self.current_positions else
            np.array([*self.current_positions[from_node] * 5, 0])
        )

        new_line = Line(
            start_pos,
            end_pos,
            # real_pos1,
            # real_pos2,
            color=color,
        )
        new_line.set_stroke(color=color)
        new_line.set_fill(color=color)
        # new_line.from_puzzle_hash = edge_to_add[0]
        # new_line.to_puzzle_hash = edge_to_add[1]
        new_line.from_node = from_node
        new_line.to_node = to_node
        self.edge_mgroups.append(new_line)

    def run_spring_layout_re_balance(self, offset_x=0, offset_y=0, seed=0, y_distance=0.3):
        graph = nx.Graph()
        for node in self.current_nodes:
            graph.add_node(node)
        for edge in self.current_edges:
            graph.add_edge(*edge)

        positions = self.current_positions
        positions[self.first_node] = np.array([0, 0])

        # y_distance = ([0.75, 0.55, 0.45, 0.3, 0.25] + [0.2] * 5 + [0.1] * 15)[i]
        # y_distance = ([0.75, 0.55, 0.45, 0.3, 0.25] + [0.2 - (i / 20) * 0.1 for i in range(20)])[i]

        max_y_distance = max(self.hashable_to_y_distance.values())

        for j in range(10):
            positions = nx.spring_layout(
                graph,
                pos=positions,  # TODO: Set the new ones close to the parents by default, not randomly
                seed=seed,
                # fixed=first_node  # First node always at the center
            )
            for hash, position in positions.items():
                node_distance = self.hashable_to_y_distance[hash]
                position[1] = -(node_distance - max_y_distance / 2) * y_distance  # + i * y_distance / 2
                position[0] += offset_x
                position[1] += offset_y

        print("run_spring_layout_re_balance", positions)
        self.current_positions = positions

    def set_positions_of_mobjects(self):
        for node in self.node_mgroups:
            three_d_pos = np.array([*self.current_positions[node.node] * 5, 0])
            node.move_to(three_d_pos)
        for line in self.edge_mgroups:
            three_d_pos_1 = np.array([*self.current_positions[line.from_node] * 5, 0])
            three_d_pos_2 = np.array([*self.current_positions[line.to_node] * 5, 0])
            line.put_start_and_end_on(three_d_pos_1, three_d_pos_2)

    def animate_position_change_of_mobjects(self) -> list[Animation]:
        animations = []
        for node in self.node_mgroups:
            three_d_pos = np.array([*self.current_positions[node.node] * 5, 0])
            animations.append(node.animate.move_to(three_d_pos))
        for line in self.edge_mgroups:
            three_d_pos_1 = np.array([*self.current_positions[line.from_node] * 5, 0])
            three_d_pos_2 = np.array([*self.current_positions[line.to_node] * 5, 0])
            animations.append(Transform(
                line,
                Line(
                    three_d_pos_1,
                    three_d_pos_2,
                    color=line.color,
                )
            ))
            # animations.append(line.animate.put_start_and_end_on(three_d_pos_1, three_d_pos_2))
        return animations

    def transition_nodes_to_balls(self):
        anims = []

        for hash, puzzle in list(self.hash_to_node_mgroup.items()):  # Need to turn into list first since we modify this
            circle = (
                Circle(color=WHITE, radius=0.5)
                .set_fill(color=BLACK, opacity=1)
                .move_to(puzzle.get_center())
                .set_z_index(10000)
            )
            if puzzle == self.hash_to_node_mgroup[self.first_node]:
                circle.set_fill(color=RED, opacity=1)
            if puzzle == self.hash_to_node_mgroup[self.goal_node]:
                circle.set_fill(color=GREEN, opacity=1)

            # anims.append(Transform(puzzle, circle))
            anims.append(FadeIn(circle))
            anims.append(FadeOut(puzzle))

            circle.node = hash
            self.hash_to_node_mgroup[hash] = circle
            self.node_mgroups[self.node_mgroups.index(puzzle)] = circle

        return anims
