import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib3d"))
from tomkit import *
from daemon_style import CYAN, EDGE, ICE, DaemonScene, label, network, plate, steps

setup(164)


class Page164(DaemonScene):
    def construct(self):
        def mesh(amplitude, color):
            def project(u,v):
                return np.array([u+.48*v,.3*v+amplitude*np.sin(u)*np.cos(v),0])
            lines = VGroup()
            for k in np.linspace(-1.5,1.5,11):
                for swap in (False,True):
                    pts = [project(k,t) if swap else project(t,k) for t in np.linspace(-1.5,1.5,25)]
                    lines.add(VMobject(stroke_color=color,stroke_width=2).set_points_as_corners(pts))
            return lines
        curves = VGroup(mesh(.9,ICE), mesh(.68,CYAN), label("TARGET  /  MODEL FIT",21,CYAN,mono=True).shift(DOWN*1.6))
        self.diagram_episode(164,"FEEDFORWARD NETWORKS",[
            ("APPROXIMATION","Chase the target.", "The cyan model surface fits\nsamples of the pale target.", curves),
            ("DIRECTION","One way through.", "During a feedforward prediction,\neach layer receives earlier values.", network([3,4,3,1])),
            ("APPLICATION","Recognize objects.", "Convolutions can build a\nfeedforward image classifier.", steps(("IMAGE","Pixels"),("FILTER","Local features"),("OUTPUT","Object scores"))),
            ("COMPOSITION","Chain functions.", "The output of one function\nbecomes the input to the next.", steps(("INPUT","x"),("FIRST","h = f₁(x)"),("NEXT","y = f₂(h)"))),
        ])
