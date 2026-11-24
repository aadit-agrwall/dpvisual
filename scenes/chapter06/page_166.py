"""Feature maps, illustrated with a two-ring classification problem."""
import json
import os
import sys
from pathlib import Path

import manimgx as m
import numpy as np

NAVY, ICE, CYAN, GOLD, EDGE = "#001A33", "#DCEFFC", "#76E0E4", "#D4A817", "#416883"


def text(words, size, color=ICE, bold=False):
    return m.Text(words, font="Noto Sans", font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


class Page166(m.Scene):
    def construct(self):
        durations = json.loads(os.environ.get("DAEMON_BEATS", "[10,12,10,13]"))
        curves = json.loads(Path(os.environ["MOTION_CURVES"]).read_text())
        times = np.linspace(0, 1, len(curves["reveal"]))
        reveal = lambda t: float(np.interp(t, times, curves["reveal"]))
        transform = lambda t: float(np.interp(t, times, curves["transform"]))
        w, h = m.config.frame_width, m.config.frame_height
        left, top = -w/2+.48, h/2-.65
        portrait = h > w
        center = np.array([0 if portrait else 3.12, -.55, 0])
        def at(x, y):
            return center + [x, y, 0]
        def left_text(words, size, y, color=ICE, bold=False):
            item = text(words, size, color, bold)
            return item.move_to([left+item.width/2, y, 0])

        name = left_text("///   TOM DAEMON", 19, top, GOLD, True)
        topic = text("FEATURE MAPS", 16, CYAN)
        topic.move_to([w/2-.48-topic.width/2, top, 0])
        self.add(name, topic, m.Line([left,top-.32,0],[-left,top-.32,0], color=EDGE))
        titles = [
            ("REPRESENTATION", "Change the view.", "Two rings overlap along x.\nTheir radii tell them apart."),
            ("GENERIC MAP", "Who is nearby?", "An RBF kernel measures similarity\nwithout listing its feature coordinates."),
            ("HANDCRAFTED MAP", "Design a feature.", "A human chooses the measurement.\nHere, radius fits the geometry."),
            ("LEARNED MAP", "Let features adapt.", "Choose the architecture. Train the\nfeature map and readout together."),
        ]
        cuts, sounds = [], []
        start = 0
        for i, (tag, title, note) in enumerate(titles):
            y = top-(1.15 if portrait else 1.42)
            words = m.VGroup(left_text(tag,18,y,CYAN), left_text(title,39,y-.65,ICE,True),
                             left_text(note,22,y-1.48))
            sounds.append([self.time,"sweep"])
            self.play(m.FadeIn(words, shift=m.UP*.12), run_time=.46, rate_func=reveal)
            objects = []
            def show(*items):
                objects.extend(items)
                sounds.append([self.time,"tick"])
                self.play(*[m.FadeIn(item, shift=m.UP*.08) for item in items], run_time=.52, rate_func=reveal)
            if i == 0:
                dots, targets = [], []
                for k,(radius,color) in enumerate([(.76,CYAN),(1.53,GOLD)]):
                    for j, angle in enumerate(np.linspace(0,2*np.pi,18,endpoint=False)):
                        r = radius + .075*np.sin(3*angle+.3)
                        dots.append(m.Dot(at(r*np.cos(angle),r*np.sin(angle)),radius=.08,color=color))
                        targets.append(at(-1.92+2.15*r, (j-8.5)*.12))
                show(m.VGroup(*dots))
                self.wait(.75)
                sounds.append([self.time,"particles"])
                self.play(*[dot.animate.move_to(pos) for dot,pos in zip(dots,targets)],
                          run_time=1.3, rate_func=transform)
                show(m.Line(at(.58,-1.5),at(.58,1.45),color=ICE),
                     text("RADIUS THRESHOLD",19,GOLD).move_to(at(.55,-1.9)))
            elif i == 1:
                pos = at(-.45,.2)
                halos = [m.Circle(radius=r,color=EDGE,stroke_width=2).move_to(pos) for r in (.46,.92,1.38)]
                show(m.VGroup(*halos),m.Dot(pos,radius=.11,color=GOLD))
                near, far = at(.06,.49), at(1.78,-.57)
                show(m.Line(pos,near,color=CYAN),m.Dot(near,radius=.09,color=CYAN),
                     text("near",21,CYAN).move_to(near+m.UP*.35))
                show(m.Dot(far,radius=.09,color=ICE),text("far",21).move_to(far+m.UP*.35))
                show(text("Closer → greater similarity",23).move_to(at(0,-1.85)))
            elif i == 2:
                c, p = at(0,.22),at(1.08,1.3)
                show(m.Circle(radius=1.53,color=EDGE).move_to(c),m.Dot(c,radius=.08,color=ICE))
                show(m.Line(c,p,color=GOLD,stroke_width=5),m.Dot(p,radius=.1,color=CYAN),
                     text("r",26,GOLD).move_to(at(.31,.99)))
                show(text("φ(x) = √(x₁² + x₂²)",30).move_to(at(0,-1.9)))
            else:
                cols = [[at(x,y) for y in np.linspace(-.76,.76,n)] if n>1 else [at(x,0)]
                        for x,n in [(-1.8,2),(0,4),(1.8,1)]]
                for k, column in enumerate(cols):
                    edges = [] if k==0 else [m.Line(a,b,color=EDGE,stroke_width=2) for a in cols[k-1] for b in column]
                    group = m.VGroup(*edges,*[m.Dot(p,radius=.135,color=GOLD if k==1 else CYAN) for p in column])
                    show(group, text(["input","φ(x; θ)","output"][k],21).move_to(at([-1.8,0,1.8][k],-1.33)))
                show(text("ŷ = φ(x; θ)ᵀw",30,GOLD).move_to(at(0,-2.03)))
            end = start + durations[i]
            self.wait(max(.04,end-self.time-.2))
            if i < 3:
                self.play(m.FadeOut(words,*objects),run_time=.2)
                cuts.append({"type":"cut","t":self.time})
            else:
                self.wait(.2)
            start = self.time
        out = Path("media/sfx")
        out.mkdir(parents=True,exist_ok=True)
        (out/"page_166_cuts.json").write_text(json.dumps(cuts))
        (out/"page_166_motions.json").write_text(json.dumps(sounds))


if __name__ == "__main__":
    mode, output = sys.argv[1:]
    m.config.pixel_width, m.config.pixel_height = (1080,1920) if mode=="portrait" else (1920,1080)
    m.config.frame_rate = 30
    m.config.background_color = NAVY
    Page166().render(output,preset="veryfast",crf=18)
