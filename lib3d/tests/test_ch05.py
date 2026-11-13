from manim import *
import numpy as np
from lib3d import *


class Test3DCh05(Stage3D):
    def construct(self):
        self.setup_stage("iso")
        report = []

        def log(line):
            report.append(line)
            open("/tmp/engine3d_ch05_test.txt", "w").write("\n".join(report) + "\n")

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

        def capacity():
            self.caption("Capacity")
            c = CapacityDemo3D(); c.reveal(self)
            c.fit(self, 1, 0.6); c.fit(self, 2, 0.6); c.fit(self, 8, 0.6); self.clear_scene()

        def linfit():
            self.caption("Linear regression")
            l = LinearFit3D(); l.reveal(self)
            l.fit(self, 5, 0.3, 0.2); l.residuals(self); self.clear_scene()

        def kfold():
            self.caption("k-fold")
            k = KFold3D(); k.reveal(self); k.cycle(self, 0.3); self.clear_scene()

        def knn():
            self.caption("kNN")
            k = KNN3D(); k.reveal(self); k.classify(self, 3); self.clear_scene()

        def pca():
            self.caption("PCA")
            p = PCA3D(); p.reveal(self); p.show_axes(self)
            p.project(self, 2, 1.0); p.unproject(self, 0.8); self.clear_scene()

        def sgd():
            self.caption("SGD")
            s = SGDLandscape(); s.reveal(self)
            s.descend(self, steps=5, step_run=0.2)
            s.descend_noisy(self, start=(-2.0, 1.5), steps=8, step_run=0.2); self.clear_scene()

        def mini():
            self.caption("Minibatch")
            m = MinibatchDemo(); m.reveal(self); self.view("top", 1.0)
            m.sweep(self, 2); self.clear_scene()

        def recipe():
            self.caption("ML recipe")
            r = MLRecipe3D(); r.reveal(self); self.view("slant", 1.0)
            r.run(self, 0.4); r.reset(self); self.clear_scene()

        def curse():
            self.caption("Curse of dimensionality")
            c = CurseOfDim3D(); c.reveal(self); c.sparse(self); self.clear_scene()

        def manifold():
            self.caption("Manifold")
            m = ManifoldDemo(); m.reveal(self); m.unroll(self); m.roll(self); self.clear_scene()

        shot("CapacityDemo3D", capacity)
        shot("LinearFit3D", linfit)
        shot("KFold3D", kfold)
        shot("KNN3D", knn)
        shot("PCA3D", pca)
        shot("SGDLandscape", sgd)
        shot("MinibatchDemo", mini)
        shot("MLRecipe3D", recipe)
        shot("CurseOfDim3D", curse)
        shot("ManifoldDemo", manifold)
        self.wait(0.5)
