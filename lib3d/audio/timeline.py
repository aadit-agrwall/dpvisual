"""Records the real animation timeline of a scene so SFX are generated from it (no guessing)."""
import json
import os

_ACC = "#1F4FFF"


def _accent(m):
    try:
        n = 0
        for s in m.get_family():
            if not s.has_points():
                continue
            n += 1
            if n > 12:
                break
            if s.get_fill_opacity() > 0.05 and s.get_fill_color().to_hex().upper() == _ACC:
                return True
            if s.get_stroke_width() > 0 and s.get_stroke_color().to_hex().upper() == _ACC:
                return True
    except Exception:
        pass
    return False


def _desc(a):
    d = {"cls": type(a).__name__}
    try:
        if d["cls"] == "_AnimationBuilder":
            oa = getattr(a, "overridden_animation", None)
            if oa is not None:
                return _desc(oa)
            m = a.mobject
            tgt = getattr(m, "target", None)
            was = _accent(m)
            will = _accent(tgt) if tgt is not None else was
            d["cls"] = "Method"
            d["state"] = "lit" if (will and not was) else ("unlit" if (was and not will) else "move")
            return d
        m = getattr(a, "mobject", None)
        ts = getattr(m, "tex_string", None)
        if ts:
            d["text"] = str(ts)
        if m is not None:
            d["n"] = len(m.submobjects) if type(m).__name__ in ("VGroup", "Group") else 1
        subs = getattr(a, "animations", None)
        if subs:
            d["n_sub"] = len(subs)
    except Exception:
        pass
    return d


class TimelineMixin:
    """Put FIRST in the bases:  class MyScene(TimelineMixin, Page142): pass"""

    def _tl(self):
        if not hasattr(self, "_timeline"):
            self._timeline = []
        return self._timeline

    def _now(self):
        try:
            return float(self.time)
        except Exception:
            return float(self.renderer.time)

    def play(self, *args, **kw):
        if not getattr(self, "_in_cam", False):
            try:
                rt = kw.get("run_time")
                if rt is None:
                    rt = max([getattr(a, "run_time", None) or 1.0 for a in args] or [1.0])
                self._tl().append({"t": self._now(), "dur": float(rt), "kinds": [_desc(a) for a in args]})
            except Exception:
                pass
        return super().play(*args, **kw)

    def move_camera(self, *a, **kw):
        self._tl().append({"t": self._now(), "dur": float(kw.get("run_time", 1.0)),
                           "kinds": [{"cls": "Camera"}]})
        self._in_cam = True
        try:
            return super().move_camera(*a, **kw)
        finally:
            self._in_cam = False

    def tear_down(self):
        super().tear_down()
        path = os.environ.get("SFX_TIMELINE", "media/sfx/timeline.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"total": self._now(), "events": self._tl()}, f)
