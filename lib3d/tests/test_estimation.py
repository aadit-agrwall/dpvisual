from manim import *
import numpy as np
from lib3d import *


class Test3DEst(Stage3D):
    def construct(self):
        self.setup_stage("iso")
        report = []

        def log(line):
            report.append(line)
            open("/tmp/engine3d_est_test.txt", "w").write("\n".join(report) + "\n")

        def shot(name, fn):
            try:
                self.view("iso", 0.3)
                fn(); log("OK    " + name)
            except Exception as e:
                log(f"FAIL  {name}: {type(e).__name__}: {e}")
                try:
                    self.remove(*self.mobjects)
                except Exception:
                    pass
                self._stage = []; self._cap = None

        def guard():
            self.view("front", 0.3)
            assert abs(self.camera.get_theta() - VIEWS["slant"][1]) < 0.05, "front was not mapped to slant"

        def bv():
            self.caption("Bias and variance")
            b = BiasVariance3D(); b.reveal(self)
            b.fit(self, 1, 0.7); b.fit(self, 8, 0.7); b.show_mean(self); self.clear_scene()

        def mle():
            self.caption("Maximum likelihood")
            m = MLE3D(); m.reveal(self); m.ascend(self, steps=10); self.clear_scene()

        def bayes():
            self.caption("Bayesian estimation")
            b = Bayesian3D(); b.reveal(self)
            for _ in range(4):
                b.update(self)
            self.clear_scene()

        def kern():
            self.caption("Kernel trick")
            k = KernelTrick3D(); k.reveal(self); k.lift(self); k.unlift(self); self.clear_scene()

        def smooth():
            self.caption("Smoothness")
            s = Smoothness3D(); s.reveal(self); s.show_bumpy(self); s.regularize(self); self.clear_scene()

        shot("view guard (front -> slant)", guard)
        shot("BiasVariance3D", bv)
        shot("MLE3D", mle)
        shot("Bayesian3D", bayes)
        shot("KernelTrick3D", kern)
        shot("Smoothness3D", smooth)
        self.wait(0.5)
