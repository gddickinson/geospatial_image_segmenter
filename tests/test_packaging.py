"""setup.py: the console-script entry point exists; PDAL is optional (pip cannot install it)."""
import ast
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _setup_kwargs():
    call = next(n for n in ast.walk(ast.parse((ROOT / "setup.py").read_text()))
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "setup")
    return {k.arg: k.value for k in call.keywords}


def test_entry_point_targets_an_existing_function():
    kw = _setup_kwargs()
    (script,) = ast.literal_eval(kw["entry_points"])["console_scripts"]
    module, func = script.split("=")[1].split(":")
    spec = importlib.util.find_spec(module)
    assert spec is not None, module
    tree = ast.parse(Path(spec.origin).read_text())
    assert any(isinstance(n, ast.FunctionDef) and n.name == func for n in tree.body)


def test_pdal_is_an_extra_not_a_requirement():
    kw = _setup_kwargs()
    reqs = ast.literal_eval(kw["install_requires"])
    assert not any(r.lower().startswith("pdal") for r in reqs)
    assert any(r.startswith("pdal") for r in ast.literal_eval(kw["extras_require"])["lidar"])


@pytest.mark.skipif(importlib.util.find_spec("laspy") is None, reason="needs laspy (desktop dependency)")
def test_lidar_utils_imports_without_pdal_and_explains_when_needed():
    code = ("import sys; sys.modules['pdal'] = None\n"
            "from geo_segmenter.utils import lidar_utils\n"
            "try:\n    lidar_utils.require_pdal()\nexcept ImportError as e:\n    print('MSG', e)\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert "MSG" in r.stdout and "conda" in r.stdout, r.stdout + r.stderr
