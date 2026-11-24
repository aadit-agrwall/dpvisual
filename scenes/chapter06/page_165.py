import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib3d"))
from tomkit import *
from daemon_style import CYAN, EDGE, ICE, DaemonScene, label, network, plate, steps

setup(165)


class Page165(DaemonScene):
    def construct(self):
        width = network([2,5,1])
        self.diagram_episode(165,"LAYERS AND UNITS",[
            ("DEPTH","Count the layers.", "Trace the chain of transformations\nfrom input to output.", network([2,3,3,2,1])),
            ("TRAINING SIGNAL","Who gets a label?", "Hidden layers learn through the\noutput error during training.", steps(("TARGET","Observed label y"),("OUTPUT","Compare prediction to y"),("HIDDEN","Adjust from gradients"))),
            ("WIDTH","Count across.", "The middle column has five units.\nEach computes one activation.", width),
            ("FEATURE MAP","Give linear models\na new input.", "An interaction can become a\nfeature in a transformed space.", steps(("INPUT","Two values x₁ and x₂"),("FEATURE","Their product x₁ × x₂"),("MODEL","Weight the new feature"))),
        ])
