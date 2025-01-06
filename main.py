from manim import *


class WaterTests(Scene):
    def construct(self):
        whole_height = 4.2

        circle = Circle(radius=0.5, color=WHITE)
        rectangle = Rectangle(height=whole_height - 0.5, width=1.0, color=WHITE)
        circle.move_to(DOWN * (whole_height / 2 - 0.25))

        bottle = Union(circle, rectangle)

        self.add(bottle)

        bottom_rectangle_pos = DOWN * (whole_height / 2 - 0.25)

        bases = [
            Square(side_length=1).set_fill(RED, 0.8),
            Square(side_length=1).set_fill(GREEN, 0.8),
            Square(side_length=1).set_fill(BLUE, 0.8),
            Square(side_length=1).set_fill(YELLOW, 0.8),
        ]

        big_clipping_mask = Rectangle(height=4, width=1.0)
        big_clipping_mask.move_to(bottom_rectangle_pos + UP * 5.5)
        big_clipping_mask.set_stroke(width=0)
        self.add(big_clipping_mask)

        def get_clipping_mask_updater(rectangle_orig):
            rectangle = rectangle_orig.copy()
            def update_clipping_mask(x):
                intersection = Intersection(Difference(rectangle, big_clipping_mask), bottle)
                x.set_points(intersection.get_points())
            return update_clipping_mask

        for i, base in enumerate(bases):
            base.move_to(bottom_rectangle_pos + UP * i)
            base.add_updater(get_clipping_mask_updater(base))
            base.set_stroke(width=0)
            self.add(base)

        self.play(big_clipping_mask.animate.shift(DOWN * 4), run_time=2)
        self.play(big_clipping_mask.animate.shift(UP* 4), run_time=2)
