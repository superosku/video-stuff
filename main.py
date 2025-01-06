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
        clipping_masks = [
            Square(side_length=1),
            Square(side_length=1),
            Square(side_length=1),
            Square(side_length=1),
        ]

        def get_clipping_mask_updater(rectangle_orig, clipping_mask):
            rectangle = rectangle_orig.copy()
            def update_clipping_mask(x):
                intersection = Intersection(Difference(rectangle, clipping_mask), bottle)
                x.set_points(intersection.get_points())
            return update_clipping_mask

        for i, (base, clipping_mask) in enumerate(zip(bases, clipping_masks)):
            base.move_to(bottom_rectangle_pos + UP * i)
            base.add_updater(get_clipping_mask_updater(base, clipping_mask))
            base.set_stroke(width=0)
            clipping_mask.move_to(base, UP)
            clipping_mask.shift(UP)
            clipping_mask.set_stroke(opacity=0)
            self.add(base)
            self.add(clipping_mask)

        self.play(clipping_masks[3].animate.shift(DOWN), run_time=0.5, rate_func=linear)
        self.play(clipping_masks[2].animate.shift(DOWN), run_time=0.5, rate_func=linear)
        self.play(clipping_masks[1].animate.shift(DOWN), run_time=0.5, rate_func=linear)
        self.play(clipping_masks[0].animate.shift(DOWN), run_time=0.5, rate_func=linear)

        self.play(clipping_masks[0].animate.shift(UP), run_time=0.5, rate_func=linear)
        self.play(clipping_masks[1].animate.shift(UP), run_time=0.5, rate_func=linear)
        self.play(clipping_masks[2].animate.shift(UP), run_time=0.5, rate_func=linear)
        self.play(clipping_masks[3].animate.shift(UP), run_time=0.5, rate_func=linear)
