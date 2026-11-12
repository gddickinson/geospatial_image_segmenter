# CLAUDE.md

@INTERFACE.md

- Python: keep files under 500 lines (pre-existing GUI files `gui/main_window.py`, `gui/layer_manager.py`,
  `gui/map_canvas.py`, `gui/paint_tool.py`, `models/cnn.py`, `data/layer.py` are over budget).
- Tests: `pytest tests` (GUI test runs offscreen in a subprocess; on the shared anaconda venv set
  `DYLD_LIBRARY_PATH` to a libsqlite for `sqlite3`). Node tests: `node geo_segmenter/web/static/<name>.test.mjs`.
- After changing the feature extractors, `display_stretch` or the RF export, run `python scripts/make_web_fixtures.py`
  and the node tests: the JS ports must keep matching Python.
- Web app: `python run_web.py` (port 8235). Static site: `python scripts/build_static.py`.
- Demo data is synthetic (`utils/synthetic_scene.py`); do not publish `test_data/sample.tif` (no source/licence recorded).
- Update INTERFACE.md and SESSION_LOG.md when the structure changes.
