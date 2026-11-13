from manim import *
import importlib
from lib3d.audio.timeline import TimelineMixin
_m = importlib.import_module("scenes.chapter05.page142_square")
class SfxRun(TimelineMixin, _m.Page142Square):
    pass
