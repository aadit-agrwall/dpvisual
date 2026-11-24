"""Tom Daemon scene furniture and isometric feature-map plates."""
import numpy as np
from manim import *
import json
import os
from pathlib import Path
from tomkit import BG, HEAD, TomScene

ICE = "#DCEFFC"
CYAN = "#76E0E4"
EDGE = "#416883"
GLASS = "#0A304A"
MONO = "Menlo"
SANS = "Helvetica Neue"


def label(words, size, color=ICE, mono=False, bold=False):
    return Text(words, font=MONO if mono else SANS, font_size=size,
                color=color, weight=BOLD if bold else NORMAL)


def anchored(item, x, y):
    return item.move_to([x + item.width / 2, y, 0])


def chrome(topic):
    w, h = config.frame_width, config.frame_height
    left, right = -w / 2 + .48, w / 2 - .48
    y = h / 2 - .62
    mark = VGroup(*[Line([left + i*.095, y-.09, 0], [left+.11+i*.095, y+.09, 0],
                         color=HEAD, stroke_width=4) for i in range(3)])
    name = anchored(label("TOM DAEMON", 18, mono=True, bold=True), left+.48, y)
    page = label(topic, 16, CYAN, mono=True)
    page.move_to([right-page.width/2, y, 0])
    return VGroup(mark, name, page, Line([left,y-.33,0], [right,y-.33,0], color=EDGE, stroke_width=1))


def copy_block(section, title, description):
    portrait = config.frame_height > config.frame_width
    left = -config.frame_width/2 + .48
    top = config.frame_height/2 - (1.72 if portrait else 2.12)
    tag = anchored(label(section, 18, CYAN, mono=True), left, top)
    heading = label(title, 48 if portrait else 44, bold=True)
    heading.next_to(tag, DOWN, buff=.23, aligned_edge=LEFT)
    note = label(description, 24)
    note.next_to(heading, DOWN, buff=.28, aligned_edge=LEFT)
    return VGroup(tag, heading, note)


def plate(mask, active=CYAN):
    """A raised square slab in parallel projection, with a 7 × 7 activation map."""
    def point(u, v):
        return np.array([u*.96-v*.63, u*.31+v*.46, 0])
    def face(vertices, fill, edge, width=1):
        return Polygon(*vertices, fill_color=fill, fill_opacity=1,
                       stroke_color=edge, stroke_width=width)
    corners = [point(x,y) for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    drop = DOWN*.12
    body = VGroup()
    for a,b in [(3,0),(0,1)]:
        body.add(face([corners[a],corners[b],corners[b]+drop,corners[a]+drop], "#08263D", EDGE))
    body.add(face(corners, GLASS, ICE, 1.8))
    cells = VGroup()
    for row, line in enumerate(mask):
        for col, value in enumerate(line):
            x, y = (col-3)*.25, (3-row)*.25
            corners_cell = [point(x+dx,y+dy) for dx,dy in [(-.106,-.106),(.106,-.106),(.106,.106),(-.106,.106)]]
            cells.add(face(corners_cell, active if value == "1" else "#103D55", "#20516A", .5))
    return VGroup(body,cells)


class DaemonScene(TomScene):
    def start_episode(self, topic):
        self.cuts = []
        self.sound_events = []
        self.durations = json.loads(os.environ.get("DAEMON_BEATS", "[]"))
        self.add(chrome(topic))

    def cue(self, kind):
        self.sound_events.append([round(self.renderer.time, 3), kind])

    def reveal(self, title):
        self.cue("sweep")
        self.play(LaggedStart(FadeIn(title[0]), FadeIn(title[1], shift=UP*.13),
                              FadeIn(title[2], shift=RIGHT*.1), lag_ratio=.32), run_time=.5)

    def finish_beat(self, group, last=False):
        index = len(self.cuts)
        start = self.cuts[-1] if self.cuts else 0
        target = self.durations[index] if self.durations else 6
        fade = 0 if last else .2
        remaining = start + target - self.renderer.time - fade
        if remaining > 1/30:
            self.wait(remaining)
        if not last:
            self.play(FadeOut(group), run_time=fade)
            self.cuts.append(round(self.renderer.time, 3))

    def save_cues(self, page):
        super().save_cues(page)
        Path(f"media/sfx/page_{page}_motions.json").write_text(json.dumps(self.sound_events))

    def diagram_episode(self, page, topic, beats):
        self.start_episode(topic)
        for i, (section, heading, note, diagram) in enumerate(beats):
            title = copy_block(section, heading, note)
            self.reveal(title)
            if diagram.width > 5.8:
                diagram.scale_to_fit_width(5.8)
            if diagram.height > 4.4:
                diagram.scale_to_fit_height(4.4)
            portrait = config.frame_height > config.frame_width
            diagram.move_to([0 if portrait else 3.05, -.5, 0])
            beat_duration = self.durations[i] if self.durations else 6
            interval = max(.4, (beat_duration - 1.3) / len(diagram))
            previous = None
            for part in diagram:
                self.cue("tick")
                self.play(FadeIn(part, shift=UP*.16), run_time=.36)
                hold = interval - .36
                if previous is not None and hold > .48:
                    start, end = previous.get_center(), part.get_center()
                    if np.linalg.norm(end-start) > .15:
                        travel = min(.65, hold)
                        self.cue("particles")
                        trails = [Line(start+dy*UP, end+dy*UP, color=HEAD, stroke_width=3)
                                  for dy in (-.06, 0, .06)]
                        self.play(*[ShowPassingFlash(trail, time_width=.3) for trail in trails], run_time=travel)
                        hold -= travel
                if hold > 1/30:
                    self.wait(hold)
                previous = part
            self.finish_beat(VGroup(title, diagram), last=i == len(beats)-1)
        self.save_cues(page)


def network(sizes):
    groups = VGroup()
    previous = None
    for i, count in enumerate(sizes):
        dots = VGroup(*[Dot(radius=.13, color=HEAD if i == len(sizes)-1 else CYAN)
                        for _ in range(count)]).arrange(DOWN, buff=.28).move_to(RIGHT*i*1.15)
        wires = VGroup()
        if previous is not None:
            wires.add(*[Line(a,b,color=EDGE,stroke_width=2) for a in previous for b in dots])
        groups.add(VGroup(wires,dots))
        previous = dots
    return groups


def steps(*entries):
    rows = VGroup()
    for key, value in entries:
        bar = RoundedRectangle(width=5.1,height=.7,corner_radius=.09,
                               stroke_color=EDGE,fill_color=GLASS,fill_opacity=1)
        left = anchored(label(key,19,HEAD,mono=True), -2.34, 0)
        text = anchored(label(value,23), -1.15, 0)
        rows.add(VGroup(bar,left,text))
    return rows.arrange(DOWN,buff=.17)
