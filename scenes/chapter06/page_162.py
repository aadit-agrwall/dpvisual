import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib3d"))
from tomkit import *
setup(162)

class Page162(TomScene):
    def construct(self):
        self.begin()
        # scene 1: part title + 3D network
        part = H("PART II", 30).move_to(UP * (config.frame_height / 2 - 0.9))
        title = VGroup(H("Deep Networks:", 46), H("Modern Practices", 46)).arrange(DOWN, buff=0.12).next_to(part, DOWN, buff=0.3)
        self.hud(part, title)
        e, L = net3d([3, 5, 5, 2], z0=-1.3)
        self.play(FadeIn(part), run_time=0.6)
        self.play(FadeIn(title), run_time=1.4)
        self.play(Create(e, lag_ratio=0.01), run_time=2.2)
        self.play(FadeIn(L), run_time=0.8)
        self.end_scene(11)
        # scene 2: working technology staircase
        head = H("Working technology", 34).move_to(UP * (config.frame_height / 2 - 0.9))
        lines = VGroup(T("Deep learning for practical applications", 24), T("Methods that already work", 24),
                       T("Used heavily in industry", 24)).arrange(DOWN, buff=0.22).next_to(head, DOWN, buff=0.35)
        self.hud(head, lines)
        s1 = slab(1.1, 0.8, -1.3, -2.6, BLU); s2 = slab(1.1, 1.6, 0.0, -2.6, BLU); s3 = slab(1.1, 2.4, 1.3, -2.6, GRN)
        self.play(FadeIn(head), FadeIn(lines[0]), FadeIn(s1, shift=OUT * 0.5), run_time=1.2)
        self.wait(1.5)
        self.play(FadeIn(lines[1]), FadeIn(s2, shift=OUT * 0.5), run_time=1.0)
        self.wait(2.0)
        self.play(FadeIn(lines[2]), FadeIn(s3, shift=OUT * 0.5), run_time=1.0)
        self.end_scene(11)
        # scene 3: feedforward network, signal flows layer by layer
        head = H("Next: the feedforward network", 32).move_to(UP * (config.frame_height / 2 - 0.9))
        caption_inset = 1.8 if config.pixel_height > config.pixel_width else 1.1
        c = T("Input → hidden layers → output", 28).move_to(DOWN * (config.frame_height / 2 - caption_inset))
        self.hud(head, c)
        e, L = net3d([3, 5, 5, 2], z0=-0.3)
        self.play(FadeIn(head), Create(e, lag_ratio=0.01), FadeIn(L), run_time=2.4)
        for layer in L:
            self.play(Indicate(layer, color=HEAD, scale_factor=1.25), run_time=0.8)
        self.play(FadeIn(c), run_time=0.8)
        self.end_scene(11, last=True)
        self.save_cues(162)
