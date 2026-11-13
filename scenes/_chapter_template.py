from manim import *
from lib3d import *          # engine + all ML components


class Demo(Stage3D):
    def construct(self):
        self.setup_stage("iso")
        self.caption("CNN forward pass")
        n = CNN3D(); n.reveal(self); n.forward(self); self.clear_scene()
