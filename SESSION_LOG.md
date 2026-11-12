# Session log

## 2026-11-12: web interface, static site and bug fixes (branch `web-interface`)
- Untracked 37 `.pyc` and 5 `.DS_Store` files; ignored the runtime log/cache and `site/`.
- Fixes (each with a regression test, separate commits):
  1. Package imports were eager (rasterio, PDAL, PyTorch, Qt needed for any module); `texture.py` imported
     `skimage.measure.shannon_entropy` (removed in scikit-image 0.26), breaking Train Model.
  2. Spectral indices wrapped around on uint8/uint16 rasters (`nir - red` unsigned): now float64.
  3. Texture "standard deviation" was the local mean; texture feature names (old GLCM list) did not match the maps;
     constant image divided by zero.
  4. FeatureSet column order depended on the hash seed, so importances could be attributed to the wrong extractor.
  5. `RandomForestModel.predict_proba` returned labels.
  6. `pytest tests` collected blocking GUI scripts: moved to `tests/gui_scripts/` (paths fixed).
- New Qt-free modules: `models/pipeline.py` (band map + extractors, used by `main_window.train_model`),
  `utils/display_stretch.py` (used by `RasterLayer.prepare_display_data`), `utils/synthetic_scene.py` (demo).
  Texture grey conversion written out instead of `np.dot` (platform-independent rounding; same values here).
- Web interface `geo_segmenter/web/` (FastAPI, port 8235) + static site: Home, Classify, Features, Method, About.
  JS ports: features (≤ 8e-16 relative error vs Python), display stretch (identical bytes), forest (exported sklearn
  forest: identical labels and probabilities; demo 65,536/65,536 pixels), GeoTIFF reader/writer (tifffile files read
  exactly; output read by tifffile and rasterio).
- Tests: `pytest tests` 34 passed; node tests 4/4 ok; screenshots `--static` and `--base` all ok, phone 0 px.
- Not used: `test_data/sample.tif` (10 m, UTM 31N, 3-band uint16; source and licence not recorded).
- Open for George: pre-existing files over 500 lines (main_window 1385, data/layer 710, map_canvas 607,
  layer_manager 597, cnn 554, paint_tool 540); `setup.py` entry point `geo_segmenter.main:main` does not exist and
  `pdal` in install_requires is not pip-installable; RandomForestModel has no seed in the GUI.

## 2026-11-12 (later): sample data, packaging, module splits, texture option (branch `web-interface`)
- Sample data: `test_data/sample.tif` (unknown source/licence) replaced by a synthetic GeoTIFF from
  `scripts/make_sample_data.py` (seed 2026; same layout: 1001 x 1001, 3 bands uint16, 10 m, EPSG:32631, same origin).
  `test_data/sample.tif.aux.xml` removed; `geo_segmenter/gui/temp/map_capture.tif` untracked and `gui/temp/`,
  `*.aux.xml`, `*.egg-info/` ignored. DATA_SOURCES.md records sources and licences. Old blobs are still in history
  (0ab471f3 sample.tif, d60482d5 sample.tif.aux.xml, b986a34b map_capture.tif): purge separately.
- setup.py fixed: console script `geo-segmenter` -> new `geo_segmenter/main.py` (root `main.py` is a wrapper); PDAL
  moved to the `lidar` extra and imported only when ground classification runs; install_requires = what the code
  imports. `pip install -e .` tested in a fresh venv; the command opens the window offscreen without PDAL.
- Split the six files over 500 lines (paint_tool, cnn, data/layer, map_canvas, layer_manager, main_window) into
  focused modules / mixins, one refactor commit each; an offscreen script (window, sample import, canvas render
  checksum, transforms, menus, training mode) gives identical output before and after.
- Fix: FeatureDialog / ResultsDialog used QWidget without importing it (NameError on construction).
- Texture option `local_mean` (config.TEXTURE_LOCAL_MEAN = False; GUI checkbox "Texture: add local mean"):
  the map the app used by mistake before the texture fix, now opt-in. `set_parameters` refreshes feature names.
- Tests: `pytest tests` 44 passed; CI=1 test_web 10 passed; clean venv with requirements-web.txt: 40 passed,
  4 skipped (desktop deps); node 4/4 ok; screenshots --static (1280 and 390 px) and --base :8235 all ok, phone 0 px.
