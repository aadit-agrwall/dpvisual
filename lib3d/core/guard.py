"""Camera guard: edge-on views ('front','side') collapse x-laid-out scenes into a line.
They are auto-mapped to 'slant' unless force=True is passed."""
from lib3d.core.engine3d import Stage3D

_EDGE_ON = ("front", "side")
_orig_view = Stage3D.view
_orig_setup = Stage3D.setup_stage


def _view(self, name, run=2.2, zoom=None, force=False):
    if name in _EDGE_ON and not force:
        name = "slant"
    return _orig_view(self, name, run, zoom)


def _setup(self, view="iso", zoom=0.85):
    if view in _EDGE_ON:
        view = "slant"
    return _orig_setup(self, view, zoom)


Stage3D.view = _view
Stage3D.setup_stage = _setup
