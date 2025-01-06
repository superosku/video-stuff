from manim import *


BOTTLE_ROTATION = -PI / 2 + PI / 16


class WaterFlask(VGroup):
    def __init__(self, colors: List[ManimColor], fill_amount: int):
        super().__init__()

        whole_height = 4.2

        circle = Circle(radius=0.5, color=WHITE)
        rectangle = Rectangle(height=whole_height - 0.5, width=1.0, color=WHITE)

        circle.move_to(DOWN * (whole_height / 2 - 0.25))

        bottle = Union(circle, rectangle)
        bottle.set_points(bottle.points[4:])  # Remove the top line from the bottle (open from top)

        self.add(bottle)

        bottom_rectangle_pos = DOWN * (whole_height / 2 - 0.25)

        self.rectangles = [
            Square(side_length=1).set_fill(color, 0.8)
            for color in colors
        ]
        self.invisble_shadow_rectangles = [
            Square(side_length=1)
            for _ in range(4)
        ]

        self.big_clipping_mask = Rectangle(height=4, width=1.0)
        self.big_clipping_mask.move_to(bottom_rectangle_pos + UP * (1.5 + fill_amount))
        self.big_clipping_mask.set_fill(PURPLE, 0.2)
        self.big_clipping_mask.set_stroke(width=0)
        # self.add(self.big_clipping_mask)

        def get_clipping_mask_updater(invisible_rectangle):
            def update_clipping_mask(x):
                intersection = Intersection(Difference(invisible_rectangle, self.big_clipping_mask), bottle)
                x.set_points(intersection.get_points())
            return update_clipping_mask

        for i, (rectangle, invisible_rectangle) in enumerate(zip(self.rectangles, self.invisble_shadow_rectangles)):
            rectangle.move_to(bottom_rectangle_pos + UP * i)
            invisible_rectangle.move_to(bottom_rectangle_pos + UP * i)
            updater_func = get_clipping_mask_updater(invisible_rectangle)
            rectangle.add_updater(updater_func)
            updater_func(rectangle)  # To have it right on the very first frame
            rectangle.set_stroke(width=0)
            invisible_rectangle.set_stroke(width=0)
            invisible_rectangle.set_fill(WHITE, 0)
            self.add(invisible_rectangle)
            self.add(rectangle)

    def animate_empty(self, n: int) -> AnimationGroup:
        local_down = rotate_vector(DOWN, BOTTLE_ROTATION)
        return AnimationGroup(
            self.big_clipping_mask.animate.shift(local_down * n),
        )

    def animate_fill(self, n: int) -> AnimationGroup:
        return AnimationGroup(
            self.big_clipping_mask.animate.shift(UP * n)
        )

    def set_colors(self, colors: List[ManimColor]):
        for base, color in zip(self.rectangles, colors):
            base.set_fill(color, 0.8)

    def set_color(self, i: int, color: ManimColor):
        self.rectangles[i].set_fill(color, 0.8)


class WaterTests(Scene):
    def construct(self):
        flask1 = WaterFlask([RED, YELLOW, YELLOW, BLUE], 2)
        flask1.shift(RIGHT * 1)
        flask1.big_clipping_mask.shift(RIGHT * 1)
        self.add(flask1.big_clipping_mask)
        self.add(flask1)

        flask2 = WaterFlask([RED, GREEN, BLUE, YELLOW], 4)
        flask2.shift(LEFT * 1)
        flask2.big_clipping_mask.shift(LEFT * 1)
        self.add(flask2)
        self.add(flask2.big_clipping_mask)

        self.wait()

        for i in range(2):
            flask2.big_clipping_mask.shift(-flask2.get_center())
            flask2.big_clipping_mask.rotate_about_origin(BOTTLE_ROTATION)
            flask2.big_clipping_mask.shift(flask2.get_center())
            flask2.big_clipping_mask.shift(UP + RIGHT)

            self.play(
                flask2.animate.shift(UP + RIGHT).rotate(BOTTLE_ROTATION)
            )

            self.play(flask2.animate_empty(1), flask1.animate_fill(1))

            flask2.big_clipping_mask.shift(-flask2.get_center())
            flask2.big_clipping_mask.rotate_about_origin(-BOTTLE_ROTATION)
            flask2.big_clipping_mask.shift(flask2.get_center())
            flask2.big_clipping_mask.shift(DOWN + LEFT)

            self.play(
                flask2.animate.shift(DOWN + LEFT).rotate(-BOTTLE_ROTATION)
            )
