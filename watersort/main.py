import math
import random

from manim import *
import networkx as nx

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
            base.set_fill(color, 0.8)

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
                else WHITE
                for i in range(4)],
                fill_amount
            )
            flask.shift_with_mask(RIGHT * flask_n * 1.5 + LEFT * 1.5 * ((color_count + 1) / 2))
            self.add(flask.big_clipping_mask)
            self.add(flask)
            flasks.append(flask)
        #
        # for flask_n in range(2):
        #     flask = WaterFlask([WHITE, WHITE, WHITE, WHITE], 0)
        #     # flask.shift_with_mask(RIGHT * (color_count + flask_n - 1) * 1.5)
        #     flask.shift_with_mask(RIGHT * (color_count + flask_n) * 1.5 + LEFT * 1.5 * ((color_count + 1) / 2))
        #     self.add(flask.big_clipping_mask)
        #     self.add(flask)
        #     flasks.append(flask)

        self.all_flasks = VGroup(*flasks)
        self.all_flasks_and_masks = VGroup(*[f.big_clipping_mask for f in flasks], self.all_flasks)
        self.flasks = flasks

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

                if y == 7 and 2 < x < 12:
                    square.set_fill(WHITE, 1.0)
                if x == 11 and 1 < y < 7:
                    square.set_fill(WHITE, 1.0)

                squares.append(square)
                squares_by_coords[(x, y)] = square
            row = VGroup(*squares)
            row.arrange(RIGHT, buff=0)
            self.rows.append(row)
        grid = VGroup(*self.rows).arrange()
        grid.arrange(DOWN, buff=0)

        grid.move_to(ORIGIN + LEFT * 1)
        grid.scale(0.5)

        self.add(grid)

        person_square = squares_by_coords[self.start_coords]
        person = Circle(radius=0.2).move_to(person_square.get_center()).set_z_index(20)
        person.set_fill(RED, 1.0)

        goal_square = squares_by_coords[self.goal_coords]
        goal = Triangle().move_to(goal_square.get_center()).set_z_index(20).scale(0.2)
        goal.set_fill(GREEN, 1.0)

        self.add(person)
        self.add(goal)

        # Path finding
        # self.wait()
        self.animate_path_finding()
        self.clear_path_finding_text()

        return

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

        self.play(
            *hide_walls_animations
        )

        self.wait()

        self.play(
            *transform_to_circles_animations,
        )

        self.wait()

        self.play(
            *[FadeIn(l) for l in connecting_lines]
        )

        # self.remove(*sum([
        #     [
        #         square
        #         for square in row
        #         if square.fill_color == WHITE
        #     ]
        #     for row in rows
        # ], []))

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

        self.play(*animations)

        self.wait()

        self.animate_path_finding()
        self.clear_path_finding_text()

        self.wait()

    def animate_path_finding(self):
        node_mobjects = []
        nodes = []
        self.node_queue = VGroup()
        self.add(self.node_queue)
        self.all_texts = []
        node_i = 0

        move_queue_top = ORIGIN + RIGHT * 5 + UP * 2.5

        def push_nodes_to_queue(node_distances: list[tuple[int, tuple[int, int]]]) -> list[Animation]:
            new_texts = []
            for distance, coords in node_distances:
                nodes.append((distance, coords))
                new_text = Text(f"{distance} ({coords[0], coords[1]})")
                node_mobjects.append(new_text)
                new_text.set_z_index(30).scale(0.5)
                new_text.align_to(self.node_queue)
                self.node_queue.add(new_text)
                new_texts.append(new_text)
            self.node_queue.arrange(DOWN, buff=0.1)
            self.node_queue.move_to(move_queue_top, aligned_edge=UP)
            return [
                FadeIn(new_text)
                for new_text in new_texts
            ]

        def pop_from_node_queue(node_i) -> tuple[int, int, tuple[int, int]]:
            distance, coords = nodes[node_i]
            node_to_be_removed = node_mobjects[node_i]

            self.node_queue.remove(node_to_be_removed)
            # node_queue.arrange(DOWN, buff=0.1)
            # node_queue.move_to(move_queue_top, aligned_edge=UP)

            self.play(
                FadeOut(node_to_be_removed),
                self.node_queue.animate
                .arrange(DOWN, buff=0.1)
                .move_to(move_queue_top, aligned_edge=UP),
                run_time=0.2
            )

            return (
                (node_i + 1, distance, coords)
            )

        push_nodes_to_queue([(0, self.start_coords)])

        for i in range(10):
            node_i, distance, current_node = pop_from_node_queue(node_i)

            if current_node == self.goal_coords:
                print("FOUND SOLUTION")
                break
            x, y = current_node

            nodes_to_push = []
            texts_to_push = []

            if x + 1 < self.width and self.rows[y][x + 1].fill_color != WHITE:
                if not any([node[1] == (x + 1, y) for node in nodes]):
                    nodes_to_push.append((x + 1, y))
                    texts_to_push.append(
                        Text(f"{distance + 1}").set_z_index(30).scale(0.5).move_to(self.rows[y][x + 1].get_center())
                    )
            if y + 1 < self.height and self.rows[y + 1][x].fill_color != WHITE:
                if not any([node[1] == (x, y + 1) for node in nodes]):
                    nodes_to_push.append((x, y + 1))
                    texts_to_push.append(
                        Text(f"{distance + 1}").set_z_index(30).scale(0.5).move_to(self.rows[y + 1][x].get_center())
                    )
            if x - 1 >= 0 and self.rows[y][x - 1].fill_color != WHITE:
                if not any([node[1] == (x - 1, y) for node in nodes]):
                    nodes_to_push.append((x - 1, y))
                    texts_to_push.append(
                        Text(f"{distance + 1}").set_z_index(30).scale(0.5).move_to(self.rows[y][x - 1].get_center())
                    )
            if y - 1 >= 0 and self.rows[y - 1][x].fill_color != WHITE:
                if not any([node[1] == (x, y - 1) for node in nodes]):
                    nodes_to_push.append((x, y - 1))
                    texts_to_push.append(
                        Text(f"{distance + 1}").set_z_index(30).scale(0.5).move_to(self.rows[y - 1][x].get_center())
                    )

            animations = [
                *push_nodes_to_queue([(distance + 1, node) for node in nodes_to_push]),
                *[FadeIn(text) for text in texts_to_push]
            ]
            self.all_texts.extend(texts_to_push)
            if animations:
                self.play(*animations, run_time=0.5)

    def clear_path_finding_text(self):
        self.play(
            FadeOut(self.node_queue),
            *[FadeOut(text) for text in self.all_texts]
        )
        self.remove(self.node_queue)
        self.remove(*self.all_texts)
        pass


class WaterSortAsGraph(Scene):
    def construct(self):
        initial_state = WaterPuzzleState.new_random(num_colors=3, random_seed=4)
        solver = WaterPuzzleSolver(initial_state)
        solver.solve()

        current_nodes = set()
        current_edges = set()
        hash_to_puzzle = {}

        desired_scale = 1.0
        positions = None
        first_node = list(solver.distance_to_hashables[0])[0]

        all_lines = []

        for i, (distance, hashables) in enumerate(sorted(solver.distance_to_hashables.items())):
            puzzles_to_add = []
            line_edges_to_add = []
            for hashable in hashables:
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

                flasks = VGroup(all_flasks).set_z_index(20)
                rect = SurroundingRectangle(
                    flasks,
                    buff=1,
                    corner_radius=1,
                    color=GREEN if puzzle.puzzle.is_solved() else WHITE,
                ).set_z_index(10).set_fill(
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

            y_distance = 0.2
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

            print("ASDF", distance, len(hashables))

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

