"""Compile every literal say()/caption()/label() string with Tex before rendering.
usage: PYTHONPATH=$PWD python3 lib3d/preflight_tex.py scenes/chapter05/page_148.py"""
import ast, sys
from manim import Tex

src = open(sys.argv[1]).read()
tree = ast.parse(src)
found = []
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
        if n.func.attr in ("say", "caption") and n.args:
            a = n.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                found.append((n.lineno, a.value))
        elif n.func.attr == "label":
            for t in ast.walk(n):
                if isinstance(t, ast.Tuple) and t.elts and isinstance(t.elts[0], ast.Constant) \
                        and isinstance(t.elts[0].value, str):
                    found.append((t.lineno, t.elts[0].value))
bad = 0
for ln, s in sorted(set(found)):
    try:
        Tex(s)
    except Exception:
        bad += 1
        print(f"REPORT: FAIL tex line {ln}: {s!r}")
print(f"REPORT: {'OK' if bad == 0 else 'FAIL'} preflight, {len(set(found))} strings checked, {bad} bad")
sys.exit(1 if bad else 0)
