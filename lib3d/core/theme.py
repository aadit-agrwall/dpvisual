"""Palette and camera views shared by the whole library."""
from manim import *

K = "#000000"
PAPER = "#FFFFFF"
WHITE_BG = PAPER
ACCENT = "#1F4FFF"   # the ONLY extra color: use it to show change (firing, selection)

VIEWS = {
    "iso":    (62 * DEGREES, -50 * DEGREES),
    "flat":   (0.0,          -90 * DEGREES),   # looks exactly like a 2D scene
    "top":    (0.01,         -90 * DEGREES),
    "side":   (90 * DEGREES, -90 * DEGREES),
    "front":  (90 * DEGREES,   0 * DEGREES),
    "back34": (68 * DEGREES, -130 * DEGREES),
    "low":    (80 * DEGREES,  -60 * DEGREES),
    "high":   (35 * DEGREES,  -60 * DEGREES),
}

# faces the x-z plane but turned away, so depth stays visible (use instead of "front"/"side")
VIEWS["slant"] = (74 * DEGREES, -70 * DEGREES)
VIEWS["oblique"] = (66 * DEGREES, -62 * DEGREES)
