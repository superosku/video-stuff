from manim import *

from watersort.helpers import WaterPuzzleState, WaterPuzzleSolver

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


class WaterPuzzleSolved(Scene):
    def construct(self):
        color_options = [YELLOW, BLUE, RED, GREEN, ORANGE, PURPLE, WHITE, GREY, PINK]
        color_count = 9
        scale_factor = 0.75
        playback_speed = 0.3

        puzzle = WaterPuzzleState.new_random(color_count)
        puzzle.print()

        solver = WaterPuzzleSolver(puzzle)
        pour_instructions = solver.solve()

        flasks = []
        for flask_n in range(color_count):
            flask = WaterFlask([color_options[puzzle.pipes[flask_n][i] - 1] for i in range(4)], 4)
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

        all_flasks = VGroup(*flasks)
        all_flasks_and_masks = VGroup(*[f.big_clipping_mask for f in flasks], all_flasks)

        self.play(
            all_flasks_and_masks.animate.scale(scale_factor, about_point=ORIGIN),
            run_time=playback_speed,
        )

        from_to = pour_instructions.pop(0)

        while True:
            _from, _to, pour_amount, pour_color, destination_empty = from_to

            flask_from = flasks[_from]
            flask_to = flasks[_to]

            move_dir = (UP * 2.5 + LEFT * 2.4 + LEFT * (_from - _to) * 1.5) * scale_factor

            flask_from.rotating = True
            self.play(
                *flask_from.move_and_rotate_animate_with_mask(move_dir, BOTTLE_ROTATION),
                run_time=playback_speed,
            )
            flask_from.rotating = False

            for i in range(pour_amount):
                flask_to.set_color(4 - destination_empty + i, color_options[pour_color - 1])

            self.play(
                flask_from.animate_empty(pour_amount, scale_factor),
                flask_to.animate_fill(pour_amount, scale_factor),
                run_time=playback_speed,
            )

            # Do not go back down while pouring from the same flask to different flasks
            # if len(pour_instructions) > 0:
            #     print("ASDF", _to, pour_instructions[0])
            while len(pour_instructions) > 0 and pour_instructions[0][0] == _from:
                _to_old = _to
                from_to = pour_instructions.pop(0)
                _from, _to, pour_amount, pour_color, destination_empty = from_to
                flask_to = flasks[_to]
                for i in range(pour_amount):
                    flask_to.set_color(4 - destination_empty + i, color_options[pour_color - 1])
                new_move = LEFT * (_to_old - _to) * 1.5 * scale_factor
                move_dir += new_move
                self.play(
                    flask_from.animate.shift(new_move),
                    flask_from.big_clipping_mask.animate.shift(new_move),
                    run_time=playback_speed,
                )
                self.play(
                    flask_from.animate_empty(pour_amount, scale_factor),
                    flask_to.animate_fill(pour_amount, scale_factor),
                    run_time=playback_speed,
                )

            flask_from.rotating = True
            self.play(
                *flask_from.move_and_rotate_animate_with_mask(-move_dir, -BOTTLE_ROTATION),
                run_time=playback_speed,
            )
            flask_from.rotating = False

            try:
                from_to = pour_instructions.pop(0)
            except IndexError:
                break

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
