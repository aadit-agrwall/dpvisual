import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib3d"))
from tomkit import *
from daemon_style import CYAN, EDGE, ICE, DaemonScene, label, network, plate, steps

setup(163)


class Page163(DaemonScene):
    def construct(self):
        seven = ["0000000", "0111110", "0000110", "0001000", "0011000", "0010000", "0100000"]
        hand = ["0000000", "1111110", "0000100", "0111110", "0010000", "0010000", "0100000"]
        parcel = VGroup(plate(seven), label("SORT TO CHUTE   →   7", 24, HEAD, mono=True)).arrange(DOWN)
        fresh = VGroup(plate(hand), label("NEW SAMPLE   →   ?", 24, HEAD, mono=True)).arrange(DOWN)
        self.diagram_episode(163, "MODERN PRACTICES", [
            ("PRACTICAL TASK", "Sort the parcel.", "A digit on the label tells us\nwhich chute to use.", parcel),
            ("SUPERVISED LEARNING", "Learn from pairs.", "Training pairs an input image\nwith its correct label.", steps(("INPUT","Pixel values"),("TARGET","Digit 7"),("UPDATE","Reduce prediction error"))),
            ("GENERALIZATION", "Read a new seven.", "A different writer crosses the seven.\nCan the network still recognize it?", fresh),
            ("THE ROADMAP", "Build, then refine.", "Choose an architecture to match\nthe structure of the input.", steps(("BUILD","Feedforward layers"),("FIT","Regularize and optimize"),("IMAGE","Convolution"),("SERIES","Recurrence"))),
        ])
