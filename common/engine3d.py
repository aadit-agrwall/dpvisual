# SHIM: moved to lib3d. New code should use: from lib3d import *
import importlib as _il
_m = _il.import_module("lib3d.core.engine3d")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
