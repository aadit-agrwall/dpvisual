"""Tom Daemon scene furniture and isometric feature-map plates."""
import numpy as np
from manim import *
from tomkit import BG, HEAD

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


def chrome(topic, page_number):
    w, h = config.frame_width, config.frame_height
    left, right = -w / 2 + .48, w / 2 - .48
    y = h / 2 - .62
    mark = VGroup(*[Line([left + i*.095, y-.09, 0], [left+.11+i*.095, y+.09, 0],
                         color=HEAD, stroke_width=4) for i in range(3)])
    name = anchored(label("TOM DAEMON", 18, mono=True, bold=True), left+.48, y)
    page = label(f"{topic}  /  {page_number}", 16, CYAN, mono=True)
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
