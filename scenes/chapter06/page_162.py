import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib3d"))
from tomkit import *
from daemon_style import CYAN, EDGE, ICE, chrome, copy_block, label, plate

setup(162)
SEVEN = ["0000000", "0111110", "0000110", "0001000", "0011000", "0010000", "0100000"]
STROKE = ["0000000", "0111110", "0000000", "0000000", "0000000", "0000000", "0000000"]
ANGLE = ["0000000", "0001110", "0000110", "0001000", "0000000", "0000000", "0000000"]
EMPTY = ["0000000"] * 7


class Page162(TomScene):
    def construct(self):
        self.cuts = []
        self.add(chrome("DEEP NETWORKS", 162))
        portrait = config.frame_height > config.frame_width
        center = RIGHT * (0 if portrait else 3.05) + DOWN * (.55 if portrait else .35)
        captions = [
            ("NETWORK DEPTH", "What makes it\ndeep?", "Count the hidden layers\nbetween pixels and prediction."),
            ("IMAGE INPUT", "Which digit\ndo you see?", "Each lit tile is one pixel\nin the input image."),
            ("FORWARD PASS", "Follow one\nprediction.", "In this example, a stroke and\na corner help identify a seven."),
        ]

        def cut(content):
            self.play(FadeOut(content), run_time=.5)
            self.cuts.append(round(self.renderer.time, 2))

        def stack():
            items = VGroup()
            for i, (mask, name) in enumerate(zip([SEVEN, STROKE, ANGLE, EMPTY],
                                                 ["INPUT", "HIDDEN 1", "HIDDEN 2", "OUTPUT"])):
                panel = plate(mask, HEAD if i in (1,2) else CYAN)
                panel.shift(center + UP*((i-1.5)*1.16) + RIGHT*((i-1.5)*.28-.47))
                anchor = panel.get_right()
                leader = Line(anchor, anchor+RIGHT*.34, color=EDGE, stroke_width=2)
                text = label(name, 17, CYAN if i in (0,3) else HEAD, mono=True)
                text.next_to(leader, RIGHT, buff=.08)
                items.add(VGroup(panel, leader, text))
            return items

        title = copy_block(*captions[0])
        layers = stack()
        self.add(title)
        for item in layers:
            self.play(FadeIn(item, shift=UP*.25), run_time=.65)
        for item in layers[1:3]:
            self.play(Indicate(item[2], color=ICE), run_time=.75)
        self.wait(6.9)
        cut(VGroup(title, layers))

        title = copy_block(*captions[1])
        photo = plate(SEVEN).scale(1.52).shift(center)
        self.add(title)
        self.play(FadeIn(photo[0]), run_time=.6)
        self.play(LaggedStart(*[FadeIn(cell) for cell in photo[1]], lag_ratio=.045), run_time=1.9)
        self.wait(3)
        result = label("YOUR READ   /   7", 27, HEAD, mono=True, bold=True)
        result.next_to(photo, DOWN, buff=.52)
        self.play(FadeIn(result, shift=UP*.12), run_time=.6)
        self.wait(4.9)
        cut(VGroup(title, photo, result))

        title = copy_block(*captions[2])
        layers = stack()
        self.add(title, layers)
        for layer in layers[1:]:
            layer.set_opacity(.27)
        self.wait(1.1)
        for prior, following in zip(layers, layers[1:]):
            a, b = prior[0].get_center(), following[0].get_center()
            # Signals travel along parallel routes between successive feature maps.
            traces = VGroup(*[Line(a+RIGHT*x, b+RIGHT*x, color=HEAD, stroke_width=5)
                              for x in (-.52,0,.52)])
            self.play(*[ShowPassingFlash(line, time_width=.55) for line in traces], run_time=1.0)
            self.play(following.animate.set_opacity(1), run_time=.45)
        answer = label("7", 46, HEAD, mono=True, bold=True).move_to(layers[-1][0])
        answer.set_background_stroke(color=BG, width=6)
        self.play(FadeIn(answer), run_time=.6)
        self.wait(4.95)
        self.save_cues(162)
