from manim import *
import networkx as nx

from watersort.helpers import WaterPuzzleState, WaterPuzzleSolver, PourInstruction

BOTTLE_ROTATION = -PI / 2 + PI / 16


class WaterFlask(VGroup):
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

    def set_color(self, i: int, color: ManimColor):
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


COLOR_OPTIONS = [YELLOW, BLUE, RED, GREEN, ORANGE, PURPLE, WHITE, GREY, PINK]


class WaterPuzzle(VGroup):
    scale_factor: float
    playback_speed: float
    color_count: int
    full_solve_instructions: list[PourInstruction]
    all_flasks: VGroup
    flasks: list[WaterFlask]

    def __init__(
        self,
        color_count: int = 9,
        playback_speed: float = 0.3,
        random_seed: int = None
    ):
        super().__init__()

        self.color_count = color_count
        self.scale_factor = 1.0
        self.playback_speed = playback_speed

        puzzle = WaterPuzzleState.new_random(self.color_count, random_seed=random_seed)

        solver = WaterPuzzleSolver(puzzle)
        self.full_solve_instructions = solver.solve()

        flasks = []
        for flask_n in range(color_count):
            flask = WaterFlask([COLOR_OPTIONS[puzzle.pipes[flask_n][i] - 1] for i in range(4)], 4)
            flask.shift_with_mask(RIGHT * flask_n * 1.5 + LEFT * 1.5 * ((color_count + 1) / 2))
            self.add(flask.big_clipping_mask)
            self.add(flask)
            flasks.append(flask)

        for flask_n in range(2):
            flask = WaterFlask([WHITE, WHITE, WHITE, WHITE], 0)
            # flask.shift_with_mask(RIGHT * (color_count + flask_n - 1) * 1.5)
            flask.shift_with_mask(RIGHT * (color_count + flask_n) * 1.5 + LEFT * 1.5 * ((color_count + 1) / 2))
            self.add(flask.big_clipping_mask)
            self.add(flask)
            flasks.append(flask)

        self.all_flasks = VGroup(*flasks)
        self.all_flasks_and_masks = VGroup(*[f.big_clipping_mask for f in flasks], self.all_flasks)
        self.flasks = flasks

    def animate_pours(self, scene: Scene, pour_instructions: list[PourInstruction] = None):
        inst = pour_instructions.pop(0)

        while True:
            flask_from = self.flasks[inst.from_]
            flask_to = self.flasks[inst.to]

            move_dir = (UP * 2.5 + LEFT * 2.4 + LEFT * (inst.from_ - inst.to) * 1.5) * self.scale_factor

            flask_from.rotating = True
            scene.play(
                *flask_from.move_and_rotate_animate_with_mask(move_dir, BOTTLE_ROTATION),
                run_time=self.playback_speed,
            )
            flask_from.rotating = False

            for i in range(inst.pour_amount):
                flask_to.set_color(4 - inst.destination_empty + i, COLOR_OPTIONS[inst.pour_color - 1])

            scene.play(
                flask_from.animate_empty(inst.pour_amount, self.scale_factor),
                flask_to.animate_fill(inst.pour_amount, self.scale_factor),
                run_time=self.playback_speed,
            )

            # Do not go back down while pouring from the same flask to different flasks
            # if len(pour_instructions) > 0:
            #     print("ASDF", _to, pour_instructions[0])
            while len(pour_instructions) > 0 and pour_instructions[0].from_ == inst.from_:
                _to_old = inst.to
                inst = pour_instructions.pop(0)
                # _from, _to, pour_amount, pour_color, destination_empty = from_to
                flask_to = self.flasks[inst.to]
                for i in range(inst.pour_amount):
                    flask_to.set_color(4 - inst.destination_empty + i, COLOR_OPTIONS[inst.pour_color - 1])
                new_move = LEFT * (_to_old - inst.to) * 1.5 * self.scale_factor
                move_dir += new_move
                scene.play(
                    flask_from.animate.shift(new_move),
                    flask_from.big_clipping_mask.animate.shift(new_move),
                    run_time=self.playback_speed,
                )
                scene.play(
                    flask_from.animate_empty(inst.pour_amount, self.scale_factor),
                    flask_to.animate_fill(inst.pour_amount, self.scale_factor),
                    run_time=self.playback_speed,
                )

            flask_from.rotating = True
            scene.play(
                *flask_from.move_and_rotate_animate_with_mask(-move_dir, -BOTTLE_ROTATION),
                run_time=self.playback_speed,
            )
            flask_from.rotating = False

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
        puzzle = WaterPuzzle(playback_speed=0.2)
        puzzle.scale_properly(0.75)
        self.add(puzzle)

        puzzle.animate_pours(self, puzzle.full_solve_instructions)

        self.wait()


class WaterPuzzleExplained(Scene):
    def construct(self):
        puzzle = WaterPuzzle(color_count=4, playback_speed=1.0, random_seed=1)
        puzzle.scale_properly(1.0)
        self.add(puzzle)

        puzzle.animate_pours(self, [PourInstruction(3, 4, 1, 2, 4)])
        self.wait(1)
        puzzle.animate_pours(self, [PourInstruction(2, 3, 1, 1, 1)])

        self.wait()


class WaterTests(Scene):
    def construct(self):
        flask1 = WaterFlask([YELLOW, YELLOW, BLUE, RED], 0)
        flask1.shift_with_mask(RIGHT * 1)
        self.add(flask1.big_clipping_mask)
        self.add(flask1)

        flask2 = WaterFlask([BLUE, YELLOW, RED, RED], 3)
        flask2.shift_with_mask(LEFT * 1)
        self.add(flask2)
        self.add(flask2.big_clipping_mask)

        for i in range(2):
            flask = WaterFlask([GREEN, BLUE, YELLOW, ORANGE], 4)
            flask.shift_with_mask(LEFT * (3 + i * 2))
            self.add(flask)
            self.add(flask.big_clipping_mask)

            flask = WaterFlask([RED, ORANGE, ORANGE, PURPLE], 4)
            flask.shift_with_mask(RIGHT * (3 + i * 2))
            self.add(flask)
            self.add(flask.big_clipping_mask)

        self.wait()

        for i in range(3):
            move_dir = UP * 2.5 + LEFT * 0.5

            flask2.change_z_indexes(20)

            flask2.rotating = True
            self.play(*flask2.move_and_rotate_animate_with_mask(move_dir, BOTTLE_ROTATION))
            flask2.rotating = False

            self.play(flask2.animate_empty(1), flask1.animate_fill(1))

            flask2.rotating = True
            self.play(*flask2.move_and_rotate_animate_with_mask(-move_dir, -BOTTLE_ROTATION))
            flask2.rotating = False

            flask2.change_z_indexes(-20)

        self.wait()


class PahtFinding(Scene):
    def construct(self):
        width = 17
        height = 10

        rows = []
        squares_by_coords = {}
        for y in range(height):
            squares = []
            for x in range(width):
                square = Square(side_length=1)
                square.set_z_index(10)
                square.set_fill(BLACK, 1.0)

                if y == 7 and 2 < x < 12:
                    square.set_fill(WHITE, 1.0)

                squares.append(square)
                squares_by_coords[(x, y)] = square
            row = VGroup(*squares)
            row.arrange(RIGHT, buff=0)
            rows.append(row)
        grid = VGroup(*rows).arrange()
        grid.arrange(DOWN, buff=0)

        grid.move_to(ORIGIN)
        grid.scale(0.5)

        self.add(grid)

        start_coords = (3, 3)
        goal_coords = (15, 8)
        person_square = squares_by_coords[start_coords]
        person = Circle(radius=0.2).move_to(person_square.get_center()).set_z_index(20)
        person.set_fill(RED, 1.0)

        goal_square = squares_by_coords[goal_coords]
        goal = Triangle().move_to(goal_square.get_center()).set_z_index(20).scale(0.2)
        goal.set_fill(GREEN, 1.0)

        self.add(person)
        self.add(goal)

        self.wait()

        # Path finding

        nodes = [(0, start_coords)]
        node_i = 0
        for i in range(1000):
            distance, current_node = nodes[node_i]
            node_i += 1
            if current_node == goal_coords:
                print("FOUND SOLUTION")
                break
            x, y = current_node

            if x + 1 < width and rows[y][x + 1].fill_color != WHITE:
                if not any([node[1] == (x + 1, y) for node in nodes]):
                    self.play(FadeIn(
                        Text(f"{distance + 1}").set_z_index(30).scale(0.5).move_to(rows[y][x + 1].get_center())
                    ), run_time=0.05)
                    nodes.append((distance + 1, (x + 1, y)))
            if y + 1 < height and rows[y + 1][x].fill_color != WHITE:
                if not any([node[1] == (x, y + 1) for node in nodes]):
                    self.play(FadeIn(
                        Text(f"{distance + 1}").set_z_index(30).scale(0.5).move_to(rows[y + 1][x].get_center())
                    ), run_time=0.05)
                    nodes.append((distance + 1, (x, y + 1)))
            if x - 1 >= 0 and rows[y][x - 1].fill_color != WHITE:
                if not any([node[1] == (x - 1, y) for node in nodes]):
                    self.play(FadeIn(
                        Text(f"{distance + 1}").set_z_index(30).scale(0.5).move_to(rows[y][x - 1].get_center())
                    ), run_time=0.05)
                    nodes.append((distance + 1, (x - 1, y)))
            if y - 1 >= 0 and rows[y - 1][x].fill_color != WHITE:
                if not any([node[1] == (x, y - 1) for node in nodes]):
                    self.play(FadeIn(
                        Text(f"{distance + 1}").set_z_index(30).scale(0.5).move_to(rows[y - 1][x].get_center())
                    ), run_time=0.05)
                    nodes.append((distance + 1, (x, y - 1)))

        # Transformation from grid to graph

        self.wait()

        connecting_lines = []
        edges = []
        connecting_lines_by_edge = {}
        for y in range(height):
            for x in range(width):
                if x + 1 < width:
                    if rows[y][x].fill_color != WHITE and rows[y][x + 1].fill_color != WHITE:
                        line = Line(
                            rows[y][x].get_center(),
                            rows[y][x + 1].get_center(),
                        )
                        line.set_z_index(5)
                        connecting_lines.append(line)
                        edges.append(((x, y), (x + 1, y)))
                        connecting_lines_by_edge[((x, y), (x + 1, y))] = line
                if y + 1 < height:
                    if rows[y][x].fill_color != WHITE and rows[y + 1][x].fill_color != WHITE:
                        line = Line(
                            rows[y][x].get_center(),
                            rows[y + 1][x].get_center(),
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
                if square.fill_color != WHITE
                else
                Transform(
                    square,
                    Dot()
                        .move_to(square.get_center())
                        .set_stroke(WHITE, opacity=0)
                        .set_fill(WHITE, opacity=0)
                )
                for square in row
            ]
            for row in rows
        ], [])

        self.play(
            *transform_to_circles_animations,
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
            new_pos[0] /= scale_factor_x / scale_extra_large_factor
            new_pos[1] /= scale_factor_y / scale_extra_large_factor
            # Pad the new pos with one 0
            new_positions[key] = np.array([*new_pos, 0])

        animations = []
        for node, new_position in new_positions.items():
            square = squares_by_coords[node]
            animations.append(square.animate.move_to(new_position))

        for y in range(height):
            for x in range(width):
                if x + 1 < width:
                    edge1 = connecting_lines_by_edge.get(((x, y), (x + 1, y)))
                    if edge1:
                        posa = new_positions[(x, y)]
                        posb = new_positions[(x + 1, y)]
                        anim = Transform(edge1, Line(posa, posb))
                        animations.append(anim)
                if y + 1 < height:
                    edge2 = connecting_lines_by_edge.get(((x, y), (x, y + 1)))
                    if edge2:
                        posa = new_positions[(x, y)]
                        posb = new_positions[(x, y + 1)]
                        anim = Transform(edge2, Line(posa, posb))
                        animations.append(anim)

        self.play(*animations)

        self.wait()
