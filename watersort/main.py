import math
import json
import random

from manim import *
import networkx as nx

from watersort.grapher import Grapher
from watersort.helpers import WaterPuzzleState, WaterPuzzleSolver, PourInstruction, HashablePuzzleState

BOTTLE_ROTATION = -PI / 2 + PI / 16


class WaterFlask(VGroup):
    rotating: bool

    def __init__(self, colors: List[ManimColor], fill_amount: int):
        super().__init__()

        # This is stored so the intersection is not calculated while the bottle is rotating
        # Doing this causes weird off by 1 frame issues with the animation
        self.rotating = False

        whole_height = 4.2

        circle = Circle(radius=0.5, color=WHITE)
        rectangle = Rectangle(height=whole_height - 0.5, width=1.0, color=WHITE)

        circle.move_to(DOWN * (whole_height / 2 - 0.25))

        self.bottle = Union(circle, rectangle)
        self.bottle.set_z_index(10)
        self.bottle.set_points(self.bottle.points[4:])  # Remove the top line from the bottle (open from top)

        self.add(self.bottle)

        bottom_rectangle_pos = DOWN * (whole_height / 2 - 0.25)

        self.rectangles = [
            Square(side_length=1).set_fill(color, 1.0).set_stroke(color).set_z_index(5-i)
            for i, color in enumerate(colors)
        ]
        self.invisble_shadow_rectangles = [
            Square(side_length=1)
            for _ in range(4)
        ]

        self.big_clipping_mask = Rectangle(height=4, width=1.0)
        self.big_clipping_mask.move_to(bottom_rectangle_pos + UP * (1.5 + fill_amount))
        self.big_clipping_mask.set_fill(PURPLE, 0.0)
        self.big_clipping_mask.set_stroke(width=0)
        # self.add(self.big_clipping_mask)

        def get_clipping_mask_updater(invisible_rectangle):
            def update_clipping_mask(x):
                if self.rotating:
                    return
                intersection = Intersection(
                    Difference(invisible_rectangle, self.big_clipping_mask), self.bottle
                )
                x.set_points(intersection.get_all_points())
            return update_clipping_mask

        for i, (rectangle, invisible_rectangle) in enumerate(zip(self.rectangles, self.invisble_shadow_rectangles)):
            rectangle.move_to(bottom_rectangle_pos + UP * i)
            invisible_rectangle.move_to(bottom_rectangle_pos + UP * i)
            updater_func = get_clipping_mask_updater(invisible_rectangle)
            rectangle.add_updater(updater_func, 0)
            updater_func(rectangle)  # To have it right on the very first frame
            # rectangle.set_stroke(width=1)
            invisible_rectangle.set_stroke(width=0)
            invisible_rectangle.set_fill(ORANGE, 0.0)
            self.add(rectangle)
            self.add(invisible_rectangle)

    def animate_empty(self, n: int, scale_factor: float) -> AnimationGroup:
        local_down = rotate_vector(DOWN, BOTTLE_ROTATION)
        return AnimationGroup(
            self.big_clipping_mask.animate.shift(local_down * n * scale_factor),
        )

    def animate_fill(self, n: int, scale_factor: float) -> AnimationGroup:
        return AnimationGroup(
            self.big_clipping_mask.animate.shift(UP * n * scale_factor)
        )

    def set_colors(self, colors: List[ManimColor]):
        for base, color in zip(self.rectangles, colors):
            base.set_fill(color)
            base.set_stroke(color)

    def set_color_of_water(self, i: int, color: ManimColor):
        self.rectangles[i].set_fill(color, 1.0)
        self.rectangles[i].set_stroke(color)

    def shift_with_mask(self, shift_vector: np.ndarray):
        self.shift(shift_vector)
        self.big_clipping_mask.shift(shift_vector)

    def move_and_rotate_animate_with_mask(self, shift_vector: np.ndarray, rotation: float) -> tuple[Mobject, ...]:
        return (
            self.animate.shift(shift_vector).rotate(rotation),
            self.big_clipping_mask.animate
                .shift(-self.get_center())
                .rotate_about_origin(rotation)
                .shift(self.get_center())
                .shift(shift_vector)
        )

    def change_z_indexes(self, amount):
        for rectangle in self.rectangles:
            rectangle.set_z_index(rectangle.get_z_index() + amount)
        self.bottle.set_z_index(self.bottle.get_z_index() + amount)


COLOR_OPTIONS = [
    YELLOW, BLUE, RED, GREEN, ORANGE, PURPLE, GREY, PINK,
    LIGHT_PINK, LIGHT_GREY, LIGHT_BROWN, GREY_BROWN, DARK_BROWN, DARK_GREY, DARKER_GREY,
    MAROON, GOLD, TEAL, TEAL_E, GREEN_A, GREEN_B, GREEN_C, GREEN_D, GREEN_E, PURE_GREEN,
    YELLOW_A, BLUE_A, RED_A, PURPLE_A, PURPLE_B, GREY_A, BLUE_B, BLUE_C, BLUE_D, WHITE,
]


class WaterPuzzle(VGroup):
    scale_factor: float
    playback_speed: float
    color_count: int
    # full_solve_instructions: list[PourInstruction]
    all_flasks: VGroup
    flasks: list[WaterFlask]
    solver: WaterPuzzleSolver
    puzzle: WaterPuzzleState

    @classmethod
    def new_random(cls, color_count: int = 9, playback_speed: float = 0.3, random_seed: int = None):
        puzzle = WaterPuzzleState.new_random(color_count, random_seed=random_seed)

        return cls(
            puzzle,
            color_count,
            playback_speed,
        )

    @classmethod
    def new_from_hashable_state(cls, hashable_state: HashablePuzzleState, playback_speed: float = 0.3):
        puzzle = WaterPuzzleState.from_hashable_state(hashable_state)
        # breakpoint()

        return cls(
            puzzle,
            len(hashable_state) - 2,
            playback_speed,
        )

    def __init__(
        self,
        puzzle: WaterPuzzleState,
        color_count: int = 9,
        playback_speed: float = 0.3,
    ):
        super().__init__()

        self.puzzle = puzzle

        self.color_count = color_count
        self.scale_factor = 1.0
        self.playback_speed = playback_speed

        solver = WaterPuzzleSolver(puzzle)
        self.solver = solver

        flasks = []

        for flask_n, pipe in enumerate(puzzle.pipes):
            fill_amount = sum([1 for c in pipe if c != 0])
            flask = WaterFlask([
                COLOR_OPTIONS[puzzle.pipes[flask_n][i] - 1]
                if puzzle.pipes[flask_n][i] != 0
                else BLACK
                for i in range(4)],
                fill_amount
            )
            flask.shift_with_mask(RIGHT * flask_n * 1.5 + LEFT * 1.5 * ((color_count + 1) / 2))
            self.add(flask.big_clipping_mask)
            self.add(flask)
            flasks.append(flask)

        self.all_flasks = VGroup(*flasks)
        self.all_flasks_and_masks = VGroup(*[f.big_clipping_mask for f in flasks], self.all_flasks)
        self.flasks = flasks

    def set_colors_from_hashable_state(self, hashable_state: HashablePuzzleState):
        for flask, pipe in zip(self.flasks, hashable_state):
            flask.set_colors([
                COLOR_OPTIONS[pipe[i] - 1]
                if pipe[i] != 0
                else BLACK # TODO: Black or white? Does it break something if I use black?
                for i in range(4)
            ])

    def get_pour_move_dir(self, from_: int, to: int) -> np.ndarray:
        return (UP * 2.5 + LEFT * 2.4 + LEFT * (from_ - to) * 1.5) * self.scale_factor

    def animate_flask_to_pour_position(
        self,
        scene: Scene,
        from_: int,
        to: int
    ):
        move_dir = self.get_pour_move_dir(from_, to)
        flask = self.flasks[from_]
        flask.rotating = True
        scene.play(
            *flask.move_and_rotate_animate_with_mask(move_dir, BOTTLE_ROTATION),
            run_time=self.playback_speed,
        )
        flask.rotating = False

    def animate_pouring(
        self, scene: Scene, inst: PourInstruction, flask_from: WaterFlask, flask_to: WaterFlask
    ):
        for i in range(inst.pour_amount):
            flask_to.set_color_of_water(4 - inst.destination_empty + i, COLOR_OPTIONS[inst.pour_color - 1])

        scene.play(
            flask_from.animate_empty(inst.pour_amount, self.scale_factor),
            flask_to.animate_fill(inst.pour_amount, self.scale_factor),
            run_time=self.playback_speed,
        )

    def animate_pours(self, scene: Scene, pour_instructions: list[PourInstruction] = None):
        inst = pour_instructions.pop(0)

        while True:
            flask_from = self.flasks[inst.from_]
            flask_to = self.flasks[inst.to]

            move_dir = self.get_pour_move_dir(inst.from_, inst.to)

            flask_from.change_z_indexes(20)
            self.animate_flask_to_pour_position(scene, inst.from_, inst.to)
            self.animate_pouring(scene, inst, flask_from, flask_to)

            # Do not go back down while pouring from the same flask to different flasks
            while len(pour_instructions) > 0 and pour_instructions[0].from_ == inst.from_:
                to_old_ = inst.to
                inst = pour_instructions.pop(0)
                flask_to = self.flasks[inst.to]

                sideways_move = LEFT * (to_old_ - inst.to) * 1.5 * self.scale_factor
                move_dir += sideways_move

                scene.play(
                    flask_from.animate.shift(sideways_move),
                    flask_from.big_clipping_mask.animate.shift(sideways_move),
                    run_time=self.playback_speed,
                )

                self.animate_pouring(scene, inst, flask_from, flask_to)

            flask_from.rotating = True
            scene.play(
                *flask_from.move_and_rotate_animate_with_mask(-move_dir, -BOTTLE_ROTATION),
                run_time=self.playback_speed,
            )
            flask_from.rotating = False
            flask_from.change_z_indexes(-20)

            try:
                inst = pour_instructions.pop(0)
            except IndexError:
                break

    def scale_properly(self, scale_factor):
        self.scale_factor *= scale_factor
        self.all_flasks_and_masks.scale(
            scale_factor, about_point=ORIGIN#, run_time=self.playback_speed
        )


class WaterPuzzleSolved(Scene):
    def construct(self):
        puzzle = WaterPuzzle.new_random(color_count=8, playback_speed=0.2 * (25 / 15), random_seed=444)
        puzzle.scale_properly(0.8)

        self.add(puzzle)

        self.wait(0.5)

        puzzle.solver.solve()
        puzzle.animate_pours(self, puzzle.solver.solve_instructions)

        self.wait(0.5)


def animate_cross_fade_in_out(scene: Scene):
    circle = Circle(radius=2).set_stroke(width=30, color=PURE_RED).set_fill(opacity=0)
    line = Line(
        circle.get_center() + (LEFT + UP) * math.sqrt(2),
        circle.get_center() + (RIGHT + DOWN) * math.sqrt(2),
        color=PURE_RED
    ).set_stroke(width=30)
    cross_thing = VGroup(circle, line).set_z_index(100)
    scene.play(FadeIn(cross_thing))
    scene.play(FadeOut(cross_thing))


class WaterPuzzleExplained(Scene):
    def construct(self):
        puzzle = WaterPuzzle.new_random(color_count=4, playback_speed=1.0, random_seed=1)
        puzzle = WaterPuzzle.new_from_hashable_state(
            (
                (1, 1, 4, 1),
                (4, 2, 2, 2),
                (4, 2, 3, 3),
                (1, 3, 4, 3),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
            ),
            playback_speed=1.0,
        )
        puzzle.scale_properly(1.0)
        self.add(puzzle)

        self.wait(2)

        # Pour to empty spot
        puzzle.animate_pours(self, [PourInstruction(2, 4, 2, 3, 4)])
        self.wait(0.5)

        # Pour over same color
        puzzle.animate_pours(self, [PourInstruction(3, 4, 1, 3, 2)])
        self.wait(4)

        # Animate pouring multiple
        puzzle.animate_pours(self, [PourInstruction(1, 2, 2, 2, 2)])
        self.wait(0.5)

        # Invalid pour over different color
        from_invalid = 1
        to_invalid = 3
        moving_flask = puzzle.flasks[from_invalid]
        moving_flask.change_z_indexes(20)
        move_dir = puzzle.get_pour_move_dir(from_invalid, to_invalid)
        puzzle.animate_flask_to_pour_position(self, from_invalid, to_invalid)

        # self.play(Indicate(moving_flask, color=RED), run_time=0.5)
        animate_cross_fade_in_out(self)

        # Animate invalid pour bottle back to position
        moving_flask.rotating = True
        self.play(
            *moving_flask.move_and_rotate_animate_with_mask(-move_dir, -BOTTLE_ROTATION),
            run_time=1.0,
        )
        moving_flask.rotating = False
        moving_flask.change_z_indexes(-20)

        self.wait(1)


class PathFinding(Scene):
    def construct(self):
        self.width = 17
        self.height = 10
        self.rows = []
        self.start_coords = (4, 4)
        self.goal_coords = (13, 8)

        squares_by_coords = {}
        for y in range(self.height):
            squares = []
            for x in range(self.width):
                square = Square(side_length=1)
                square.set_z_index(10)
                square.set_fill(BLACK, 1.0)
                square.is_wall = False

                if y == 7 and 2 < x < 12:
                    square.is_wall = True
                    # square.set_fill(WHITE, 1.0)
                if x == 11 and 1 < y < 7:
                    square.is_wall = True
                    # square.set_fill(WHITE, 1.0)

                squares.append(square)
                squares_by_coords[(x, y)] = square
            row = VGroup(*squares)
            row.arrange(RIGHT, buff=0)
            self.rows.append(row)
        grid = VGroup(*self.rows).arrange()
        grid.arrange(DOWN, buff=0)

        # grid.move_to(ORIGIN + LEFT * 1)
        grid.move_to(ORIGIN)
        grid.scale(0.5)

        self.wait(5)
        self.play(FadeIn(Text("Path finding").move_to(UP * 3.5).scale(0.9)))
        self.wait(2)
        self.play(FadeIn(Text("Breath First Search (BFS)").move_to(UP * 3.0).scale(0.60)))
        self.wait(5.5)
        self.play(FadeIn(grid))
        self.wait(1)

        person_square = squares_by_coords[self.start_coords]
        person = Circle(radius=0.2).move_to(person_square.get_center()).set_z_index(20)
        person.set_fill(RED, 1.0)
        self.person = person

        goal_square = squares_by_coords[self.goal_coords]
        goal = Triangle().move_to(goal_square.get_center()).set_z_index(20).scale(0.2)
        goal.set_fill(GREEN, 1.0)
        self.goal = goal

        self.play(FadeIn(goal))
        self.wait(1.5)
        self.play(FadeIn(person))
        self.wait(1)
        self.play(*[
            square.animate.set_fill(WHITE)
            for row in grid
            for square in row
            if square.is_wall
        ])
        self.wait(3)

        # Path finding
        # self.wait()
        self.animate_path_finding(slow_start=True)
        self.clear_path_finding_text()

        # return  # TODO: Temporary to speedup animation

        # Transformation from grid to graph

        # self.wait()

        connecting_lines = []
        edges = []
        connecting_lines_by_edge = {}
        for y in range(self.height):
            for x in range(self.width):
                if x + 1 < self.width:
                    if self.rows[y][x].fill_color != WHITE and self.rows[y][x + 1].fill_color != WHITE:
                        line = Line(
                            self.rows[y][x].get_center(),
                            self.rows[y][x + 1].get_center(),
                        )
                        line.set_z_index(5)
                        connecting_lines.append(line)
                        edges.append(((x, y), (x + 1, y)))
                        connecting_lines_by_edge[((x, y), (x + 1, y))] = line
                if y + 1 < self.height:
                    if self.rows[y][x].fill_color != WHITE and self.rows[y + 1][x].fill_color != WHITE:
                        line = Line(
                            self.rows[y][x].get_center(),
                            self.rows[y + 1][x].get_center(),
                        )
                        line.set_z_index(5)
                        connecting_lines.append(line)
                        edges.append(((x, y), (x, y + 1)))
                        connecting_lines_by_edge[((x, y), (x, y + 1))] = line

        transform_to_circles_animations = sum([
            [
                Transform(
                    square,
                    Circle(radius=square.width / (2 + 1))
                        .move_to(square.get_center())
                        .set_stroke(WHITE)
                        .set_fill(BLACK, 1)
                )
                for square in row
                if square.fill_color != WHITE
            ]
            for row in self.rows
        ], [])

        hide_walls_animations = sum([
            [
                Transform(
                    square,
                    Dot()
                    .move_to(square.get_center())
                    .set_stroke(WHITE, opacity=0)
                    .set_fill(WHITE, opacity=0)
                )
                for square in row
                if square.fill_color == WHITE
            ]
            for row in self.rows
        ], [])

        self.wait(9)

        self.play(
            *hide_walls_animations
        )

        self.wait(1.5)

        self.play(
            *transform_to_circles_animations,
        )

        self.wait(1.5)

        self.play(
            *[FadeIn(l) for l in connecting_lines]
        )

        graph = nx.Graph()
        for edge in edges:
            _from, _to = edge
            graph.add_edge(_from, _to)
        new_positions = nx.spring_layout(
            graph,
            pos={
                node: (node[0], -node[1])
                for node in graph.nodes
            },
            iterations=50
        )
        all_new_positions = np.array(list(new_positions.values()))
        pos_x_min, pos_y_min = all_new_positions.min(axis=0)
        pos_x_max, pos_y_max = all_new_positions.max(axis=0)

        all_old_positions = np.array([
            [square.get_center()[0], square.get_center()[1]]
            for square in squares_by_coords.values()]
        )
        old_x_min, old_y_min = all_old_positions.min(axis=0)
        old_x_max, old_y_max = all_old_positions.max(axis=0)

        scale_factor_x = (pos_x_max - pos_x_min) / (old_x_max - old_x_min)
        scale_factor_y = (pos_y_max - pos_y_min) / (old_y_max - old_y_min)

        scale_extra_large_factor = 1.2

        for key, new_pos in new_positions.items():
            # Scale the new pos
            new_pos[0] /= 0.8 * scale_factor_x / scale_extra_large_factor
            new_pos[0] -= 2  # Move slightly to the left
            new_pos[1] /= 0.8 * scale_factor_y / scale_extra_large_factor
            # Pad the new pos with one 0

            # Make new pos be the average of new and old pos
            new_pos[0] = (new_pos[0] + 2 * squares_by_coords[key].get_center()[0]) / 3
            new_pos[1] = (new_pos[1] + 2 * squares_by_coords[key].get_center()[1]) / 3

            new_positions[key] = np.array([*new_pos, 0])

        animations = []
        for node, new_position in new_positions.items():
            square = squares_by_coords[node]
            animations.append(square.animate.move_to(new_position))

        animations.append(goal.animate.move_to(new_positions[self.goal_coords]))
        animations.append(person.animate.move_to(new_positions[self.start_coords]))

        for y in range(self.height):
            for x in range(self.width):
                if x + 1 < self.width:
                    edge1 = connecting_lines_by_edge.get(((x, y), (x + 1, y)))
                    if edge1:
                        posa = new_positions[(x, y)]
                        posb = new_positions[(x + 1, y)]
                        anim = Transform(edge1, Line(posa, posb))
                        animations.append(anim)
                if y + 1 < self.height:
                    edge2 = connecting_lines_by_edge.get(((x, y), (x, y + 1)))
                    if edge2:
                        posa = new_positions[(x, y)]
                        posb = new_positions[(x, y + 1)]
                        anim = Transform(edge2, Line(posa, posb))
                        animations.append(anim)

        self.wait(4)

        self.play(*animations)

        self.wait(2)

        self.animate_path_finding()
        self.clear_path_finding_text()

        self.wait()

    def animate_path_finding(self, slow_start=False):
        nodes = []
        self.all_texts = []
        node_i = 0

        def push_nodes_to_queue(node_distances: list[tuple[int, tuple[int, int]]]) -> list[Animation]:
            for distance, coords in node_distances:
                nodes.append((distance, coords))

        def pop_from_node_queue(node_i) -> tuple[int, int, tuple[int, int]]:
            distance, coords = nodes[node_i]

            return (
                (node_i + 1, distance, coords)
            )

        def get_text_to_push(distance, coords):
            center = self.rows[coords[1]][coords[0]].get_center()
            return VGroup(
                Text(f"{distance}").set_z_index(30).scale(0.5).move_to(center),
                Circle(radius=0.2, color=BLUE).move_to(center).set_z_index(30)
            )

        all_anims = []

        start_text = get_text_to_push(0, self.start_coords)
        self.all_texts.append(start_text)
        all_anims.append([FadeIn(start_text)])

        push_nodes_to_queue([(0, self.start_coords)])

        for _ in range(500):
            text_to_fade_out = self.all_texts[node_i]
            node_i, distance, current_node = pop_from_node_queue(node_i)

            if current_node == self.goal_coords:
                assert 0  # This should not happen
                print("FOUND SOLUTION")
                break
            x, y = current_node

            nodes_to_push = []
            texts_to_push = []

            if x + 1 < self.width and self.rows[y][x + 1].fill_color != WHITE:
                if not any([node[1] == (x + 1, y) for node in nodes]):
                    nodes_to_push.append((x + 1, y))
                    texts_to_push.append(get_text_to_push(distance + 1, (x + 1, y)))
            if y + 1 < self.height and self.rows[y + 1][x].fill_color != WHITE:
                if not any([node[1] == (x, y + 1) for node in nodes]):
                    nodes_to_push.append((x, y + 1))
                    texts_to_push.append(get_text_to_push(distance + 1, (x, y + 1)))
            if x - 1 >= 0 and self.rows[y][x - 1].fill_color != WHITE:
                if not any([node[1] == (x - 1, y) for node in nodes]):
                    nodes_to_push.append((x - 1, y))
                    texts_to_push.append(get_text_to_push(distance + 1, (x - 1, y)))
            if y - 1 >= 0 and self.rows[y - 1][x].fill_color != WHITE:
                if not any([node[1] == (x, y - 1) for node in nodes]):
                    nodes_to_push.append((x, y - 1))
                    texts_to_push.append(get_text_to_push(distance + 1, (x, y - 1)))

            asdf = [(distance + 1, node) for node in nodes_to_push]
            push_nodes_to_queue(asdf)

            animations = [
                *[FadeIn(text) for text in texts_to_push],
                text_to_fade_out[1].animate.set_stroke(opacity=0)  # Do not fade out but set the opacity to fix a bug
            ]
            self.all_texts.extend(texts_to_push)
            all_anims.append(animations)

            to_break = False
            for _, node in asdf:
                if node == self.goal_coords:
                    print("FOUND SOLUTION")
                    to_break = True

            if to_break:
                break

        all_sub_anims = [AnimationGroup(*anims) for anims in all_anims if anims]
        slow_and_fast_split = 6

        # The slow part at the beginning
        if slow_start:
            self.wait(4)
            self.play(all_sub_anims[0])
            self.wait(14)
            self.play(all_sub_anims[1])
            self.wait(8)
            self.play(all_sub_anims[2])
            self.wait(1)
            self.play(all_sub_anims[3])
            self.wait(1)
            self.play(all_sub_anims[4])
            self.wait(1)
            self.play(all_sub_anims[5])
            self.wait(1)

        # The fast part
        self.play(AnimationGroup(
            all_sub_anims[slow_and_fast_split:] if slow_start else all_sub_anims,
            run_time=8,
            lag_ratio=0.2,
        ))

        win_distance, (win_x, win_y) = [n for n in nodes if n[1] == self.goal_coords][0]
        cur_dist, cur_x, cur_y = win_distance, win_x, win_y

        path = []

        for i in range(100):
            path.append((cur_x, cur_y))
            close_to_wins = sorted([
                (d, (x, y)) for d, (x, y) in nodes
                if cur_x - 1 <= x <= cur_x + 1  # Next to this x wise
                and cur_y - 1 <= y <= cur_y + 1  # Next to this y wise
                and abs(cur_x - x) + abs(cur_y - y) == 1  # But not diagonal
            ])
            cur_dist, (cur_x, cur_y) = close_to_wins[0]
            if (cur_x, cur_y) == self.start_coords:
                path.append((cur_x, cur_y))
                break

        self.wait(4 if slow_start else 1)

        # Animate moving of the player towards the goal
        path_mobjects = [self.rows[y][x] for x, y in path]
        self.play(AnimationGroup(
            *[Indicate(square, color=WHITE) for square in path_mobjects],
            run_time=8 if slow_start else 2,
            lag_ratio=0.2,
            rate_func=rate_functions.ease_in_cubic if slow_start else rate_functions.linear,
        ))

        path_centers = [self.rows[y][x].get_center() for x, y in path]
        thing = Rectangle().set_points_smoothly(
            path_centers
        ).set_fill(BLUE, opacity=0).set_stroke(PURPLE, opacity=1, width=5).set_z_index(1000000)
        # self.play(Create(thing), run_time=4)

        self.play(MoveAlongPath(self.goal, thing), run_time=4 if slow_start else 2)

        self.wait()

        self.play(FadeOut(thing), FadeOut(self.goal))
        self.goal.move_to(self.rows[self.goal_coords[1]][self.goal_coords[0]].get_center())
        self.play(FadeIn(self.goal))

        self.wait()

    def clear_path_finding_text(self):
        self.wait(0.5)
        self.play(
            FadeOut(text) for text in self.all_texts
        )
        self.wait(0.5)
        self.all_texts = []
        pass


class WaterSortAsGraph(Scene):
    def construct(self):
        # initial_state = WaterPuzzleState.new_random(num_colors=5, random_seed=4)  # TODO: Slow to render but final
        # initial_state = WaterPuzzleState.new_random(num_colors=4, random_seed=4)  # TODO: Slow to render but final
        initial_state = WaterPuzzleState.new_random(num_colors=3, random_seed=4)  # TODO: Slow to render but final
        # initial_state = WaterPuzzleState.new_random(num_colors=2, random_seed=4)  # TODO: For debug, faster to render
        solver = WaterPuzzleSolver(initial_state)
        solver.solve()

        current_nodes = set()
        current_edges = set()
        hash_to_puzzle = {}

        desired_scale = 1.0
        positions = None
        first_node = list(solver.distance_to_hashables[0])[0]

        all_lines = []

        z_index_counter = 0

        path_nodes = solver.get_all_nodes_from_start_to(solver.winning_node())

        for i, (distance, hashables) in enumerate(sorted(solver.distance_to_hashables.items())):
            puzzles_to_add = []
            line_edges_to_add = []
            for hashable in hashables:
                z_index_counter += 1
                current_nodes.add(hashable)
                current_edges.update([
                    edge
                    for edge in solver.edges
                    if edge[1] == hashable
                ])
                hashable_original_pipes = solver.hashable_to_original_unsorted[hashable].pipes
                puzzle = WaterPuzzle.new_from_hashable_state(hashable_original_pipes)
                all_flasks = puzzle.all_flasks
                for flask in all_flasks:
                    flask.rotating = True

                desired_z_index = z_index_counter * 50 + 100000 if hashable in path_nodes else 0

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
                hash_to_puzzle[hashable] = both

                for edge in current_edges:
                    # TODO: Should add direction to edges? Some arrows perhaps?
                    if edge[0] == hashable and edge[1] in hash_to_puzzle:
                        # TODO: Does this fix my issues? Does it add duplicate edges?
                        # TODO: HMM maybe it is bi directional if both are added?
                        line_edges_to_add.append(edge)
                    if edge[1] == hashable and edge[0] in hash_to_puzzle:
                        line_edges_to_add.append(edge)

            initial_positions = positions if positions is not None else {}
            for node in current_nodes:
                if node not in initial_positions:
                    start_nodes_for_this_node = [
                        e[0]
                        for e in current_edges
                        if e[1] == node and e[0] in initial_positions
                    ]
                    if not start_nodes_for_this_node:
                        initial_positions[node] = np.array([0, 0])
                    else:
                        initial_positions[node] = (
                                initial_positions[start_nodes_for_this_node[0]]
                                + np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * 0.01
                        )

            graph = nx.Graph()
            for node in current_nodes:
                graph.add_node(node)
            for edge in current_edges:
                graph.add_edge(*edge)

            positions = initial_positions
            positions[first_node] = np.array([0, 0])

            # y_distance = ([0.75, 0.55, 0.45, 0.3, 0.25] + [0.2] * 5 + [0.1] * 15)[i]
            y_distance = ([0.75, 0.55, 0.45, 0.3, 0.25] + [0.2 - (i / 20) * 0.1 for i in range(20)])[i]
            for j in range(10):
                positions = nx.spring_layout(
                    graph,
                    pos=positions,  # TODO: Set the new ones close to the parents by default, not randomly
                    # fixed=first_node  # First node always at the center
                )
                for hash, position in positions.items():
                    hash_distance = solver.hashable_to_original_unsorted[hash].distance
                    position[1] = -hash_distance * y_distance + i * y_distance / 2

            desired_scale_old = desired_scale
            desired_scale = (1 / (i + 1)) * 0.5

            hash_to_new_pos = {}

            anims = []
            for hashed, position in positions.items():
                if hashed in hash_to_puzzle:
                    to_be_moved = hash_to_puzzle[hashed]
                    new_pos = np.array([*position * 5, 0])
                    hash_to_new_pos[hashed] = new_pos
                    if to_be_moved in puzzles_to_add:
                        hash_to_puzzle[hashed].move_to(new_pos).scale(desired_scale)
                    else:
                        anims.append(
                            to_be_moved.animate
                                .move_to(new_pos)
                                .scale(desired_scale / desired_scale_old)
                        )

            for line in all_lines:
                anims.append(
                    Transform(
                        line,
                        Line(
                            hash_to_new_pos[line.from_puzzle_hash],
                            hash_to_new_pos[line.to_puzzle_hash],
                            color=WHITE,
                        )
                    )
                )

            lines_to_add = []
            for edge_to_add in line_edges_to_add:
                new_line = Line(
                    hash_to_new_pos[edge_to_add[0]],
                    hash_to_new_pos[edge_to_add[1]],
                    color=WHITE,
                )
                new_line.from_puzzle_hash = edge_to_add[0]
                new_line.to_puzzle_hash = edge_to_add[1]
                lines_to_add.append(new_line)

            if anims:
                self.play(*anims)

            self.play(FadeIn(*puzzles_to_add, *lines_to_add))

            all_lines += lines_to_add

            if i == 0:
                self.wait(7)
            if i == 1:
                self.wait(4)

            print("ASDF", distance, len(hashables))

        # Fade out all not in path
        fade_out_anims = []
        for hash, puzzle in hash_to_puzzle.items():
            if hash not in path_nodes:
                fade_out_anims.append(puzzle.animate.set_stroke(opacity=0.3).set_fill(opacity=0.3))
                # TODO: Should set the fill separately to different parts of the flasks? Ie. not on the masks and such
        for line in all_lines:
            if line.from_puzzle_hash not in path_nodes or line.to_puzzle_hash not in path_nodes:
                fade_out_anims.append(line.animate.set_stroke(opacity=0.3))
        self.play(*fade_out_anims)

        # Zoom in
        self.play(
            *[p.animate.scale(2.0) for k, p in hash_to_puzzle.items() if k in path_nodes],
            run_time=2
        )

        # Zoom to path and follow it
        all_vgroup = VGroup(
            *[p for p in hash_to_puzzle.values()],
            *all_lines
        )
        zoom_in_factor = 3.2
        all_vgroup.scale(zoom_in_factor)
        todo_center = hash_to_puzzle[path_nodes[-1]].get_center()
        all_vgroup.scale(1/zoom_in_factor)
        self.play(all_vgroup.animate.scale(zoom_in_factor).shift(-todo_center))

        path_to_follow = [
            -hash_to_puzzle[hash].get_center() - todo_center
            for hash in reversed(path_nodes)
        ]
        thing = Rectangle().set_points_smoothly(
            path_to_follow
        ).shift(-(path_to_follow[0] - all_vgroup.get_center()))
        self.wait(0.5)

        # self.play(FadeIn(thing), run_time=0.5)
        # self.play(FadeIn(Dot().move_to(all_vgroup.get_center())))
        # self.wait(0.5)

        # self.play(MoveAlongPath(all_vgroup, thing), run_time=6) #, rate_func=linear)
        self.play(MoveAlongPath(all_vgroup, thing), run_time=6, rate_func=rate_functions.ease_in_out_sine)

        # Wait
        self.wait(0.1)


class HarderAndHarder(Scene):
    def construct(self):
        puzzle1 = WaterPuzzle(WaterPuzzleState.new_random(num_colors=2, random_seed=4))
        puzzle2 = WaterPuzzle(WaterPuzzleState.new_random(num_colors=4, random_seed=4))
        puzzle3 = WaterPuzzle(WaterPuzzleState.new_random(num_colors=8, random_seed=4))
        puzzle4 = WaterPuzzle(WaterPuzzleState.new_random(num_colors=16, random_seed=4))
        puzzle5 = WaterPuzzle(WaterPuzzleState.new_random(num_colors=len(COLOR_OPTIONS), random_seed=4))

        puzzles = [
            puzzle1, puzzle2, puzzle3, puzzle4, puzzle5
        ]
        puzzle_scales = [
            1.0, 0.9, 0.8, 0.5, 0.25
        ]

        puzzle_centers = [

        ]

        for puzzle, scale in zip(puzzles, puzzle_scales):
            for flask in puzzle.all_flasks:
                flask.rotating = False
            puzzle.scale(scale)
            # puzzle_centers.append(ORIGIN)
            puzzle_centers.append(
                ORIGIN - (puzzle.all_flasks.get_center() - puzzle.all_flasks_and_masks.get_center()) * scale
            )

        off_screen = 8
        run_time = 1
        wait_time = 1

        for puzzle, center in zip(puzzles, puzzle_centers):
            puzzle.move_to(center + DOWN * off_screen)

        self.play(puzzle1.animate.move_to(puzzle_centers[0]), run_time=run_time)

        self.wait(wait_time)

        for i, ((p1, c1), (p2, c2)) in enumerate(zip(
            zip(puzzles[1:], puzzle_centers[1:]),
            zip(puzzles[:-1], puzzle_centers[:-1]),
        )):
            for flask in p1.all_flasks:
                flask.rotating = False
            for flask in p2.all_flasks:
                flask.rotating = False
            self.play(p1.animate.move_to(c1), p2.animate.move_to(c2 + UP * off_screen), run_time=run_time)
            # Last round wait more
            if i < len(puzzles) - 2:
                self.wait(wait_time)
            else:
                self.wait(3)

        self.play(puzzles[-1].animate.move_to(puzzle_centers[-1] + UP * off_screen), run_time=run_time)

        self.wait()


class GettingStuck(Scene):
    def construct(self):
        puzz = WaterPuzzleState.new_random(num_colors=8, random_seed=123)
        puzzle = WaterPuzzle(puzz)
        solver = puzzle.solver
        solver.solve()

        stuck_nodes_by_distance = sorted(solver.nodes_and_distance_that_have_no_moves, key=lambda x: x[0])
        old_puzzle = None

        for n in [0, 5, 21]:
            puzz = WaterPuzzleState.new_random(num_colors=8, random_seed=123)
            puzzle = WaterPuzzle(puzz)

            pour_instructions = solver.get_pour_instructions_into(stuck_nodes_by_distance[n][1].hashable())

            scale = 0.8
            puzzle.scale_properly(scale)
            puzzle.move_to(
                ORIGIN - (puzzle.all_flasks.get_center() - puzzle.all_flasks_and_masks.get_center()) * scale
            )
            puzzle.shift(UP * 8)
            for flask in puzzle.flasks:
                flask.rotating = True
            if old_puzzle is not None:
                self.play(
                    puzzle.animate.shift(DOWN * 8),
                    old_puzzle.animate.shift(DOWN * 8)
                )
            else:
                self.play(puzzle.animate.shift(DOWN * 8))
            for flask in puzzle.flasks:
                flask.rotating = False

            puzzle.animate_pours(self, pour_instructions)

            animate_cross_fade_in_out(self)

            for flask in puzzle.flasks:
                flask.rotating = True
            old_puzzle = puzzle
            # self.play(puzzle.animate.shift(DOWN * 8))


def load_data_by_size(file_name="output.json"):
    data_by_size = {}
    with open(f"watersort/watersort_rust/{file_name}") as f:
        for line in f.readlines():
            if not line:
                continue
            data = json.loads(line)
            data_by_size[data["size"]] = data

    print("data_by_size.keys()", data_by_size.keys())

    return data_by_size


class SolvabilityGraph(Scene):
    def construct(self):
        data_by_size = load_data_by_size()

        # width = 4
        # height = 3
        width = 5
        height = 10

        all_boths = []

        color_count_to_sample = 16

        colors_to_plot = 40
        puzzles_to_sample = 50

        solvable_at_color = sum([t["solvable"] for t in data_by_size[color_count_to_sample]["puzzles"][:puzzles_to_sample]])

        for y in range(height):
            for x in reversed(range(width)):
                data = data_by_size[color_count_to_sample]["puzzles"][x * height + y]
                pipes = data["pipes"]
                puzzle = WaterPuzzle.new_from_hashable_state(pipes)
                for flask in puzzle.flasks:
                    flask.rotating = True
                puzzle.pipedata = data
                all_flasks = puzzle.all_flasks
                all_flasks.scale(0.08)
                for f in all_flasks:
                    f.bottle.set_stroke(width=1.0)  # Set the bottle stroke
                    for r in f.rectangles:
                        r.set_stroke(width=1.0)
                # all_flasks.set_stroke(width=0.8)
                surrounding_rect = SurroundingRectangle(all_flasks, buff=0.10, corner_radius=0.05, color=WHITE).set_stroke(width=1.0)
                both = VGroup(
                    all_flasks,
                    surrounding_rect.set_z_index(-1000)
                )
                both.puzzle = puzzle
                all_boths.append(both)
                # both = all_flasks
                both.move_to(
                    ORIGIN +
                    1.0 * (2.6 * (RIGHT * x + LEFT * (width / 2 - 0.5))) +
                    1.0 * (0.75 * (UP * y + DOWN * (height / 2 - 0.5)))
                )
                # self.add(both)

        self.play(AnimationGroup(
            *[
                GrowFromPoint(both, ORIGIN)
                for both in reversed(all_boths)
            ],
            run_time=6,
            lag_ratio=0.2,
        ))

        self.wait(0.1)

        color_change_anims = []
        for both in reversed(all_boths):
            if both.puzzle.pipedata["solvable"]:
                color_change_anims.append(AnimationGroup(
                    Flash(both[0], color=WHITE),
                    both[1].animate.set_fill(GREEN, opacity=0.4),
                ))
            else:
                color_change_anims.append(AnimationGroup(
                    Flash(both[0], color=WHITE),
                    both[1].animate.set_fill(RED, opacity=0.4),
                ))

        self.play(AnimationGroup(
            *color_change_anims,
            run_time=6,
            lag_ratio=0.2,
        ))

        self.wait(1.0)

        text = MathTex(
            r"\frac{" +
            str(solvable_at_color) +
            r"}{" +
            str(puzzles_to_sample) +
            r"} = " +
            str(int(100 * solvable_at_color/puzzles_to_sample)) +
            r"\%"
        )

        all_boths_vgroup = VGroup(*all_boths)

        self.play(
            ShrinkToCenter(all_boths_vgroup),
            GrowFromPoint(text, ORIGIN),
        )
        # self.play(Transform(all_boths_vgroup, text))
        # text = all_boths_vgroup

        self.wait(1)

        ax = Axes(
            x_range=(4, colors_to_plot),  # 30 colors
            y_range=(0, 1),  # 50 puzzles
            tips=False
        )
        # labels = ax.get_axis_labels(x_label="Colors", y_label="Solvable puzzles")
        ax.add_coordinates(
            range(4, colors_to_plot + 1, 2),
            [0, 1]
        )

        self.play(Write(ax))

        self.wait(1)

        dot = Dot(ax.c2p(color_count_to_sample, solvable_at_color / puzzles_to_sample))
        text.color_count = color_count_to_sample
        self.play(Transform(text, dot))

        self.wait(1)

        all_dots = [text]
        anims = []
        for x in range(4, colors_to_plot + 1):
            if x == color_count_to_sample:
                continue

            y = sum([t["solvable"] for t in data_by_size[x]["puzzles"][0:puzzles_to_sample]]) / puzzles_to_sample  # Only 50 first puzzles

            dot = Dot(ax.c2p(x, y))
            anims.append(FadeIn(dot))
            dot.color_count = x
            all_dots.append(dot)

        self.play(
            AnimationGroup(
                *anims,
                run_time=6,
                lag_ratio=0.2,
            ),
        )

        self.wait(1)

        # Add more samples
        anims = []
        for x in range(4, colors_to_plot):
            dot_to_move = next(d for d in all_dots if d.color_count == x)
            new_y = sum([t["solvable"] for t in data_by_size[x]["puzzles"]]) / len(data_by_size[x]["puzzles"])
            anims.append(dot_to_move.animate.move_to(ax.c2p(x, new_y)))
        self.play(*anims)

        self.wait(1)

        self.play(FadeOut(ax, *all_dots))

        self.wait(1)


class PlottingNodesAndEdges(Scene):
    def construct(self):
        data_by_size = load_data_by_size()

        node_edges = [(i["nodes"], i["edges"]) for i in data_by_size[4]["puzzles"]]
        node_max, edge_max = np.max(np.array(node_edges), 0)

        ax = Axes(
            x_range=(0, node_max),  # 30 colors
            y_range=(0, edge_max),  # 50 puzzles
            tips=True,
            axis_config={"include_numbers": False, "include_ticks": False},
        )

        self.add(ax)

        dot_colors = {
            4: WHITE,
            5: RED,
            6: GREEN,
            7: BLUE,
            8: YELLOW,
        }

        all_dots = []

        anims = []
        for node, edge in node_edges:
            dot = Dot(ax.c2p(node, edge)).set_color(dot_colors[4])
            dot.my_node = node
            dot.my_edge = edge
            anims.append(FadeIn(dot))
            all_dots.append(dot)

        self.play(AnimationGroup(
            *anims,
            run_time=3,
            lag_ratio=0.2,
        ))

        for colors in range(5, 9):
            node_edges = [(i["nodes"], i["edges"]) for i in data_by_size[colors]["puzzles"]]
            node_max, edge_max = np.max(np.array(node_edges), 0)
            print("ASDF", colors, node_max, edge_max)

            # This ax is not added to the scene but just to get the coordinates
            ax = Axes(
                x_range=(0, node_max),  # 30 colors
                y_range=(0, edge_max),  # 50 puzzles
                tips=True,
                axis_config={"include_numbers": False, "include_ticks": False},
            )

            self.play(
                ax.animate.set(x_range=(0, node_max), y_range=(0, edge_max)),
                *[
                    dot.animate.move_to(ax.c2p(dot.my_node, dot.my_edge))
                    for dot in all_dots
                ],
                run_time=3
            )
            self.wait(1)

            anims = []
            for node, edge in node_edges:
                dot = Dot(ax.c2p(node, edge)).set_color(dot_colors[colors])
                dot.my_node = node
                dot.my_edge = edge
                all_dots.append(dot)
                anims.append(FadeIn(dot))

            self.play(AnimationGroup(
                *anims,
                run_time=3,
                lag_ratio=0.2,
            ))
            self.wait(1)


def add_histogram(is_x_axis: bool, ax: Axes, items: list[int], bucket_count: int = 10):
    assert min(items) >= 0
    assert max(items) <= 1

    buckets = [0 for _ in range(bucket_count)]
    for dot in items:
        bucket = int(dot * bucket_count)
        if bucket == len(buckets):
            bucket -= 1
        buckets[bucket] += 1

    max_bucket = max(buckets)

    all_rects = []
    for i, bucket in enumerate(buckets):
        ratio = bucket / max_bucket

        if is_x_axis:
            diff_between_0_and_1 = ax.c2p(0, 0)[1] - ax.c2p(0, 1)[1]
        else:
            diff_between_0_and_1 = ax.c2p(0, 0)[0] - ax.c2p(1, 0)[0]

        bar_height = ratio * 2
        bar_width = (1.0 if is_x_axis else 0.25) * 2 * diff_between_0_and_1 / bucket_count

        if is_x_axis:
            bar_coords = ax.c2p(
                i / bucket_count,
                0,
            ) - RIGHT * (bar_width / 2)
            bar_coords[1] += bar_height / 2
        else:
            bar_coords = ax.c2p(
                0,
                i / bucket_count,
            ) - UP * (bar_width / 2)
            bar_coords[0] += bar_height / 2

        rect = Rectangle(
            width=bar_width if is_x_axis else bar_height,
            height=bar_height if is_x_axis else bar_width,
            fill_opacity=0.5,
            fill_color=WHITE,
        ).move_to(bar_coords)
        all_rects.append(rect)
    return VGroup(*all_rects)


class PlottingNodesAndSolvableNodes(Scene):
    def construct(self):
        data_by_size = load_data_by_size()

        node_solvables = [
            (i["nodes"], i["winnable_nodes"])
            for i in data_by_size[14]["puzzles"]
            if i["solvable"]
        ]
        node_max, winnable_nodes_max = np.max(np.array(node_solvables), 0)

        ax = Axes(
            x_range=(0, node_max),  # 30 colors
            y_range=(0, winnable_nodes_max),  # 50 puzzles
            tips=True,
            axis_config={"include_numbers": False, "include_ticks": False},
        )

        self.add(ax)

        # Add dots
        anims = []
        all_dots = []
        for node, solvable in node_solvables:
            dot = Dot(ax.c2p(node, solvable)) #.set_color(dot_colors[4])
            dot.my_node = node
            dot.my_solvable = solvable
            anims.append(
                AnimationGroup(
                    GrowFromCenter(dot),
                    rate_func=rate_functions.ease_out_elastic,
                )
            )
            all_dots.append(dot)

        self.play(AnimationGroup(
            *anims,
            run_time=3,
            lag_ratio=0.2,
        ))

        new_ax = Axes(
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Transform dots to bottom (TODO: It is not the best to transform them like this since the whole representation changes)
        anims = []
        for dot in all_dots:
            new_dot = Dot(
                new_ax.c2p(
                    dot.my_solvable / dot.my_node,
                    0
                )
            )
            anims.append(Transform(dot, new_dot))

        self.play(*anims)

        # Add buckets
        hist = add_histogram(True, new_ax, [dot.my_solvable / dot.my_node for dot in all_dots], 10)
        self.play(FadeIn(hist))

        self.wait()


class DefiningHardness(Scene):
    def construct(self):
        title = Text("Defining difficulty")
        title.move_to(ORIGIN + UP * 3)
        self.add(title)

        initial_state = WaterPuzzleState.from_hashable_state([
            (1, 1, 2, 2),
            (2, 2, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ])
        solver = WaterPuzzleSolver(initial_state)
        solver.solve()

        first_node_hashable = list(solver.distance_to_hashables[0])[0]
        grapher = Grapher(first_node_hashable, solver.winning_node())

        for distance, nodes in solver.distance_to_hashables.items():
            for hashable in nodes:
                grapher.add_node_to_graph(solver, hashable, distance)

        for node1, node2 in solver.edges:
            grapher.add_edge_to_graph(node1, node2)

        grapher.run_spring_layout_re_balance(offset_x=-0.4, offset_y=-0.2, seed=2, y_distance=0.25)

        self.play(FadeIn(*grapher.node_mgroups, *grapher.edge_mgroups))
        self.wait(1)
        self.play(*grapher.animate_position_change_of_mobjects())
        self.wait(1)
        self.play(*grapher.transition_nodes_to_balls())

        # self.play(*grapher.transition_nodes_to_balls())

        invalids_hashables = []
        for node, distance in [
            ([k for k, v in grapher.hashable_to_y_distance.items() if v == 1][0], 1),
            ([k for k, v in grapher.hashable_to_y_distance.items() if v == 2][1], 2),
        ]:
            new_node = random.randint(0, 1000000)
            invalids_hashables.append(new_node)
            grapher.add_node_to_graph(
                None,
                new_node,
                distance + 1,
                use_circle=True,
                start_at=node,
                outline_color=RED,
            )
            grapher.add_edge_to_graph(node, new_node, color=RED)

        grapher.run_spring_layout_re_balance(offset_x=-0.4, offset_y=-0.2, y_distance=0.25, seed=2)

        # Make the newly added a tiny bit tighter
        stuff = sorted([(k, v) for k, v in grapher.current_positions.items()], key=lambda x: x[1][0])
        print(
            "WTF",
            grapher.current_positions[stuff[0][0]][0],
            grapher.current_positions[stuff[-1][0]][0],
        )
        grapher.current_positions[stuff[0][0]] = grapher.current_positions[stuff[0][0]] + np.array([0.3, 0])
        grapher.current_positions[stuff[-1][0]] = grapher.current_positions[stuff[-1][0]] - np.array([0.3, 0])

        self.play(*grapher.animate_position_change_of_mobjects())
        self.wait(1)

        is_wins = []
        hilight_node_in_win_animations = []
        for i in range(20):
            cur_node = grapher.first_node
            hilightables = []
            while True:
                hilightables.append(cur_node)
                node_options = [e[1] for e in grapher.current_edges if e[0] == cur_node]
                if len(node_options) == 0 or cur_node == grapher.goal_node:
                    break
                random_node = random.choice(node_options)
                cur_node = random_node
            is_win = cur_node == grapher.goal_node
            is_wins.append(is_win)
            hilight_node_in_win_animations.append(
                AnimationGroup(
                    *[Indicate(grapher.hash_to_node_mgroup[hashable]) for hashable in hilightables],
                    lag_ratio=0.1
                )
            )

        texts = [
            Text("1. Length of shortest path (4)"),
            Text("2. Number of nodes (9)"),
            Text("3. Number of edges (11)"),
            Text("4. Ratio of dead end nodes (2 / 9) = 78%"),
            Text(f"5. Random play win probability\n   ({sum(is_wins)} / {len(is_wins)}) = {int(100 * sum(is_wins) / len(is_wins))}%)"),
        ]
        hilight_animss = [
            [AnimationGroup(
                Indicate(grapher.node_mgroups[0]),
                Indicate(grapher.node_mgroups[1]),
                Indicate(grapher.node_mgroups[5]),
                Indicate(grapher.node_mgroups[6]),
            )],
            [AnimationGroup(*[Indicate(node) for node in grapher.node_mgroups])],
            [AnimationGroup(*[Indicate(edge) for edge in grapher.edge_mgroups])],
            [
                AnimationGroup(*[
                    Indicate(grapher.hash_to_node_mgroup[hashable])
                    for hashable in invalids_hashables
                ]),
                AnimationGroup(*[Indicate(node) for node in grapher.node_mgroups]),
            ],
            hilight_node_in_win_animations,
        ]
        for i, (text, anims) in enumerate(zip(texts, hilight_animss)):
            text.scale(0.5)
            text.move_to(ORIGIN + UP * 1.8 + i * DOWN * 0.5 + RIGHT * 1.1, aligned_edge=LEFT + UP)
            self.play(FadeIn(text))
            for anim in anims:
                self.play(anim)

        self.wait(1)


class VisualizingHardness(Scene):
    def construct(self):
        data_by_size = load_data_by_size("output2.json")

        hist_bucket_count = 10

        # Axes from x 0 to 1 and y 0 to 1
        ax = Axes(
            x_range=(0, 1),
            y_range=(0, 1),
            axis_config={"include_numbers": True, "include_ticks": True},
        )

        my_puzzles = data_by_size[8]["puzzles"]  # 50 samples

        puzzle_data = [
            [p["moves_to_reach_winnable"] for p in my_puzzles],
            [p["nodes"] for p in my_puzzles],
            [p["edges"] for p in my_puzzles],
            [1 - p["winnable_nodes_vs_nodes"] for p in my_puzzles],
            [p["random_play_wins"] for p in my_puzzles],
        ]
        puzzle_max_values = [max(p) for p in puzzle_data]
        puzzle_min_values = [min(p) for p in puzzle_data]
        puzzle_data_scaled = [
            [
                (p - min_val) / (max_val - min_val)
                # (
                #     (p / (max_val + 1))
                #     - (min_val / (max_val + 1))
                # ) * (max_val / (max_val - min_val))
                for p in pp
            ]
            for pp, max_val, min_val in zip(puzzle_data, puzzle_max_values, puzzle_min_values)
        ]
        puzzle_axis_titles = [
            "Length of shortest path",
            "Number of nodes",
            "Number of edges",
            "Ratio of dead end nodes",
            "Random play win probability",
        ]
        # breakpoint()

        current_data_x = 1
        current_data_y = 2

        def new_title_for_x(text) -> Mobject:
            return (
                Text(text).scale(0.5)
                .move_to(ax.c2p(0.5, 0) + DOWN * 0.3)
            )

        def new_title_for_y(text) -> Mobject:
            return (
                Text(text).scale(0.5)
                .move_to(ax.c2p(0, 0.5) + LEFT * 0.3)
                .rotate(PI / 2)
            )

        # Set axis labels
        ax.add_coordinates(
            {
                0: puzzle_min_values[current_data_x],
                1: puzzle_max_values[current_data_x]
            },
            {
                0: puzzle_min_values[current_data_y],
                1: puzzle_max_values[current_data_y]
            },
        )
        # Set axis titles
        ax_labels = VGroup(
            new_title_for_x(puzzle_axis_titles[current_data_x]),
            new_title_for_y(puzzle_axis_titles[current_data_y])
        )

        self.play(FadeIn(ax, ax_labels))
        self.wait()

        hilighted_dot = None
        index_of_closest = 0

        def dot_updater(dd):
            if dd == hilighted_dot and dd.has_been_scaled is False:
                dd.has_been_scaled = True
                dd.scale(2)
            if dd != hilighted_dot and dd.has_been_scaled is True:
                dd.has_been_scaled = False
                dd.scale(0.5)

        dots = []

        def add_dots(a, b):
            new_dots = []
            for i, (n, e) in enumerate(list(zip(puzzle_data_scaled[current_data_x], puzzle_data_scaled[current_data_y]))[a:b]):
                dot = Dot(ax.c2p(n, e))
                dot.has_been_scaled = False
                dot.add_updater(dot_updater)
                dot.graph_x_pos = n
                dot.graph_y_pos = e
                dot.index_in_data = a + i
                dots.append(dot)
                new_dots.append(dot)
            self.play(FadeIn(*new_dots))

        add_dots(0, 20)

        self.wait(1)

        x_diff = (ax.c2p(1, 0) - ax.c2p(0, 0))[0]
        y_diff = (ax.c2p(0, 1) - ax.c2p(0, 0))[1]
        angle_to_rotate = math.atan(y_diff / x_diff)
        show_offer_oval = Circle()
        show_offer_oval.set_points([[p[0], p[1] / 6, p[2]] for p in show_offer_oval.get_points()])
        show_offer_oval.scale(6)
        show_offer_oval.rotate(angle_to_rotate)

        # probe_pos = ax.c2p(0.0, 0.0)
        probe_pos = show_offer_oval.get_points()[0]
        preview_center = ax.c2p(0.25, 0.5)

        probe_shower = Dot(probe_pos)
        probe_line = Line(probe_pos, preview_center).set_z_index(50)

        preview_puzzle = WaterPuzzle.new_from_hashable_state(my_puzzles[0]["pipes"])
        all_flasks = preview_puzzle.all_flasks
        for flask in preview_puzzle.flasks:
            flask.change_z_indexes(150)
            flask.rotating = True
        all_flasks.scale(0.3)
        surrounding_rect = (
            SurroundingRectangle(all_flasks, buff=0.10, corner_radius=0.05, color=WHITE)
            .set_fill(BLACK, opacity=1)
            .set_z_index(100)
            # .set_stroke(width=1.0)
        )
        surr_and_flasks_and_probe_line = VGroup(all_flasks, surrounding_rect, probe_line)

        show_offer = probe_shower
        # show_offer = VGroup(probe_shower, probe_line)
        surr_and_flasks_and_probe_line.shift(UP * 2.2 + LEFT * 3.2)
        #
        # self.add(surr_and_flasks)

        def flasks_updater(vgroup):
            nonlocal index_of_closest
            puzz_to_use = my_puzzles[index_of_closest]
            pipes = puzz_to_use["pipes"]
            preview_puzzle.set_colors_from_hashable_state(pipes)
            probe_line.set_points_smoothly(
                [probe_shower.get_center(), surrounding_rect.get_center()]
            )
            # probe_line, probe_shower
            # breakpoint()
            # surr_and_flasks.move_to(probe_line.get_points()[-1])
            # surr_and_flasks.move_to(show_offer.get_center() + UP * 3)

        self.play(FadeIn(show_offer))

        flasks_updater(surr_and_flasks_and_probe_line)
        surr_and_flasks_and_probe_line.add_updater(flasks_updater, call_updater=True)

        def probe_updater(ff):
            # nonlocal hilight_dot
            # nonlocal un_hilight_dot
            nonlocal hilighted_dot
            nonlocal index_of_closest

            pos_in_graph = ax.p2c(probe_shower.get_center())
            index_of_closest = sorted([
                ((d.graph_x_pos - pos_in_graph[0]) ** 2 + (d.graph_y_pos - pos_in_graph[1]) ** 2, i)
                for i, d in
                enumerate(dots)]
            )[0][1]
            # x_in_graph = pos_in_graph[0]
            # try:
            #     index_of_closest = [i > x_in_graph for i in number_of_nodes_scaled].index(True)
            # except ValueError:
            #     # breakpoint()
            #     print("THIS HAPPENED")
            #     # TODO: What to do (Should be the last index?)
            #     index_of_closest = len(number_of_nodes_scaled) - 1

            dot_to_hilight = dots[index_of_closest]
            hilighted_dot = dot_to_hilight

        probe_shower.add_updater(probe_updater)

        x_diff_whole_graph = (ax.c2p(1, 0) - ax.c2p(0, 0))[0]
        y_diff_whole_graph = (ax.c2p(0, 1) - ax.c2p(0, 0))[1]

        self.play(FadeIn(surr_and_flasks_and_probe_line))

        self.play(
            MoveAlongPath(show_offer, show_offer_oval),
            # show_offer.animate.shift(RIGHT * x_diff_whole_graph * 1.0 + UP * y_diff_whole_graph * 1.0),
            run_time=10,
            rate_func=linear
        )

        self.wait(1)

        add_dots(20, -1)

        self.wait(1)

        self.play(
            MoveAlongPath(show_offer, show_offer_oval),
            # show_offer.animate.shift(-RIGHT * x_diff_whole_graph * 1.0 - UP * y_diff_whole_graph * 1.0),
            run_time=10,
            rate_func=linear
        )

        self.wait(1)

        self.play(FadeOut(surr_and_flasks_and_probe_line, probe_shower))

        x_hist = add_histogram(True, ax, puzzle_data_scaled[current_data_x], bucket_count=hist_bucket_count)
        self.play(FadeIn(x_hist))

        y_hist = add_histogram(False, ax, puzzle_data_scaled[current_data_y], bucket_count=hist_bucket_count)
        self.play(FadeIn(y_hist))

        self.wait(1)

        def number_as_text(number) -> Mobject:
            if isinstance(number, float):
                return MathTex(f"{int(number * 100)}\%").scale(0.8)
            string_number = str(number)
            mobject = MathTex(string_number)
            mobject.scale({
                1: 1,
                2: 1,
                3: 0.9,
                4: 0.8,
                5: 0.7,
                6: 0.6,
            }.get(len(string_number), 0.6) * 0.8)
            return mobject

        def change_dot_axis(data_x, data_y):
            nonlocal current_data_x
            nonlocal current_data_y
            current_data_x = data_x
            current_data_y = data_y

            # Dot locations
            anims = []
            for dot in dots:
                dot_index = dot.index_in_data
                data_a = puzzle_data_scaled[current_data_x][dot_index]
                data_b = puzzle_data_scaled[current_data_y][dot_index]
                anims.append(dot.animate.move_to(
                    ax.c2p(data_a, data_b)
                ))

            # Axis titles
            anims.append(ReplacementTransform(
                ax_labels[0],
                new_title_for_x(puzzle_axis_titles[current_data_x])
            ))
            anims.append(ReplacementTransform(
                ax_labels[1],
                new_title_for_y(puzzle_axis_titles[current_data_y])
            ))

            # Axis labels
            # Hight
            anims.append(ReplacementTransform(
                ax.coordinate_labels[0][1],
                number_as_text(puzzle_max_values[current_data_x])
                .move_to(ax.coordinate_labels[0][1].get_center())
            ))
            anims.append(ReplacementTransform(
                ax.coordinate_labels[1][1],
                number_as_text(puzzle_max_values[current_data_y])
                .move_to(ax.coordinate_labels[1][1].get_center())
            ))
            # Low
            anims.append(ReplacementTransform(
                ax.coordinate_labels[0][0],
                number_as_text(puzzle_min_values[current_data_x])
                .move_to(ax.coordinate_labels[0][0].get_center())
            ))
            anims.append(ReplacementTransform(
                ax.coordinate_labels[1][0],
                number_as_text(puzzle_min_values[current_data_y])
                .move_to(ax.coordinate_labels[1][0].get_center())
            ))

            # Histograms
            x_hist_new = add_histogram(True, ax, puzzle_data_scaled[current_data_x], bucket_count=hist_bucket_count)
            y_hist_new = add_histogram(False, ax, puzzle_data_scaled[current_data_y], bucket_count=hist_bucket_count)
            anims.append(Transform(x_hist, x_hist_new))
            anims.append(Transform(y_hist, y_hist_new))

            self.play(*anims)

        change_dot_axis(1, 0)
        self.wait(1)
        change_dot_axis(1, 2)
        self.wait(1)
        change_dot_axis(1, 3)
        self.wait(1)
        change_dot_axis(1, 4)
        self.wait(1)


class MutatingPuzzle(Scene):
    def construct(self):
        data_by_size = load_data_by_size("output2.json")
        my_data = data_by_size[8]["puzzles"]

        with open("watersort/watersort_rust/output_mutate.json") as f:
            lines = f.readlines()
            mutate_data = [json.loads(line) for line in lines]

        current_puzzle = WaterPuzzle.new_from_hashable_state(mutate_data[0]["pipes"])
        current_puzzle.scale_properly(0.35)
        current_puzzle.move_to(ORIGIN + LEFT * 3.5)

        self.play(FadeIn(current_puzzle))

        self.wait(1)

        texts = [
            "Length of shortest path",
            "Number of nodes",
            "Number of edges",
            "Ratio of dead end nodes",
            "Random play win probability",
        ]
        max_vals = [
            max([d["moves_to_reach_winnable"] for d in my_data]),
            max([d["nodes"] for d in my_data]),
            max([d["edges"] for d in my_data]),
            max([1 - d["winnable_nodes_vs_nodes"] for d in my_data]),
            max([d["random_play_wins"] for d in my_data]),
        ]
        min_vals = [
            min([d["moves_to_reach_winnable"] for d in my_data]),
            min([d["nodes"] for d in my_data]),
            min([d["edges"] for d in my_data]),
            min([1 - d["winnable_nodes_vs_nodes"] for d in my_data]),
            min([d["random_play_wins"] for d in my_data]),
        ]
        cur_vals = [
            mutate_data[0]["moves_to_reach_winnable"],
            mutate_data[0]["nodes"],
            mutate_data[0]["edges"],
            1 - mutate_data[0]["winnable_nodes_vs_nodes"],
            mutate_data[0]["random_play_wins"],
        ]

        axes = [
            Axes(
                x_range=(0, 1),
                y_range=(0, 1),
            ).scale(0.25)
            for _ in range(5)
        ]

        scale_x = 5.0
        scale_y = 2.2

        axes[0].move_to(ORIGIN + UP * 1.0 * scale_y + RIGHT * 0.3 * scale_x)
        axes[1].move_to(ORIGIN + UP * 0.0 * scale_y + RIGHT * 0.3 * scale_x)
        axes[2].move_to(ORIGIN + UP * -1.0 * scale_y + RIGHT * 0.3 * scale_x)

        axes[3].move_to(ORIGIN + UP * 0.5 * scale_y + RIGHT * 1.0 * scale_x)
        axes[4].move_to(ORIGIN + UP * -0.5 * scale_y + RIGHT * 1.0 * scale_x)

        # self.play(FadeIn(*axes))
        graph_rect_titles = [
            Text(text)
            .scale(0.45)
            .move_to(ax.get_center() + UP * 0.99)
            for text, ax in zip(texts, axes)
        ]

        self.play(FadeIn(*axes, *graph_rect_titles))

        last_dots = []
        for mut_index, mutated in enumerate(mutate_data):
            if not mutated["is_improvenment"]:
                continue

            current_puzzle.set_colors_from_hashable_state(mutated["pipes"])

            cur_vals = [
                mutated["moves_to_reach_winnable"],
                mutated["nodes"],
                mutated["edges"],
                1 - mutated["winnable_nodes_vs_nodes"],
                mutated["random_play_wins"],
            ]

            new_dots = []
            new_lines = []
            min_factors = [min_vals[0] - 3, min_vals[1], min_vals[2], 0, 0]
            max_factors = [max_vals[0] + 3, max_vals[1] * 3, max_vals[2] * 3, 0.2, 10000]
            for (
                ax, val, max_val, min_val
            ) in zip(axes, cur_vals, max_factors, min_factors):
                scaled_x = mut_index / len(mutate_data)
                scaled_y = (val - min_val) / (max_val - min_val)
                # (p - min_val) / (max_val - min_val)
                dot = Dot(ax.c2p(scaled_x, scaled_y), radius=DEFAULT_DOT_RADIUS * 0.5)
                new_dots.append(dot)
                if last_dots:
                    for a, b in zip(last_dots, new_dots):
                        new_lines.append(Line(a.get_center(), b.get_center(), color=WHITE))
            last_dots = new_dots

            self.play(FadeIn(*new_dots, *new_lines), run_time=0.2)

        self.wait(1)
