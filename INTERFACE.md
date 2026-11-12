# INTERFACE.md: navigation map

Geospatial Image Segmenter: a PyQt6 desktop app for segmenting/classifying geospatial imagery (raster, vector, LIDAR)
with a random forest or a CNN, plus a web interface (FastAPI, port 8235) and a static GitHub Pages site for the
classical random-forest pipeline.

## Entry points
| File | Purpose |
|---|---|
| `main.py` | starts the desktop app (`gui.main_window.MainWindow`) |
| `run_web.py` | starts the web app on http://127.0.0.1:8235 |
| `scripts/build_static.py` | exports the static site to `site/` (GET endpoints as JSON) |
| `scripts/make_sample_data.py` | writes `test_data/sample.tif` (synthetic 3-band uint16 GeoTIFF, UTM 31N, 10 m; see DATA_SOURCES.md) |
| `scripts/make_web_fixtures.py` | writes the Python-vs-JS fixtures in `tests/fixtures/` |
| `scripts/screenshots.py` | Playwright smoke test of every route + README screenshots (`docs/img/`) |

## Package `geo_segmenter/` (subpackages import lazily, see `__init__.py`)
| Module | Contents |
|---|---|
| `config.py` | paths, map URLs, SPECTRAL_INDICES, TEXTURE_WINDOWS, RF_N_ESTIMATORS / RF_MAX_DEPTH, CNN and GUI defaults |
| `models/base.py` | `GeospatialModel` (feature set, `prepare_training_data`) |
| `models/random_forest.py` | `RandomForestModel` (sklearn): `train` (80/20 stratified split, metrics, importances), `predict`, `predict_proba`, save/load |
| `models/cnn.py` | `CNNModel` (PyTorch), desktop only |
| `models/pipeline.py` | `default_band_map(n_bands)`, `add_default_extractors(model, n_bands, band_map)`: the GUI's training setup |
| `models/features/base.py` | `FeatureExtractor`, `FeatureSet` (`enabled_names`, `enabled_feature_names`: column order) |
| `models/features/spectral.py` | `SpectralFeatureExtractor`: NDVI, NDWI, SAVI, EVI (ARVI, SIPI), RG / NIR ratios |
| `models/features/texture.py` | `TextureFeatureExtractor`: local std, range, Gaussian gradient magnitude per window |
| `models/features/terrain.py` | `TerrainFeatureExtractor` for elevation data (slope, curvature, TPI, ...) |
| `data/layer.py` | `Layer`, `RasterLayer` (display via `utils/display_stretch`), vector, LIDAR, segmentation, training-label layers |
| `data/label.py`, `data/providers/` | labels; raster, tile, LIDAR and GPS providers |
| `utils/display_stretch.py` | `prepare_display(data, band_indices, stretch_params)`: band combination + stretch (numpy only) |
| `utils/synthetic_scene.py` | `make_scene(size, seed)`: seeded synthetic 4-band scene, known classes, example strokes (web demo) |
| `utils/geo_utils.py`, `image_utils.py`, `lidar_utils.py`, `logger.py` | coordinate / raster / LIDAR helpers, logging |
| `gui/main_window.py` | `MainWindow`: menus, import, labels, `train_model`, `run_segmentation` |
| `gui/*.py`, `gui/dialogs/*.py` | map canvas/view/widget, layer manager, paint tool, selection; label, stretch, band combo, RF/CNN, training, results, feature dialogs |

## Web interface `geo_segmenter/web/`
| File | Contents |
|---|---|
| `app.py` | FastAPI: `GET /api/info`, `GET /api/demo` (in `STATIC_ENDPOINTS`), `POST /api/classify` (Python engine) |
| `classify_service.py` | `run_pipeline` (pipeline + RandomForestModel), `export_forest`, base64 / packbits helpers |
| `demo.py` | cached demo: synthetic scene + Python forest (random_state 0), `demo_payload`, `demo_summary` |
| `static/index.html`, `main.js`, `core.js`, `style.css`, `style_app.css`, `icon.svg` | shell, router, theme |
| `static/home.js`, `classify.js`, `results.js`, `features_page.js`, `docs.js` | pages: home, classify workbench, results/exports, feature maps, method + about |
| `static/state.js`, `viewer.js`, `worker.js`, `worker_client.js`, `image_io.js` | session state, canvas + brush, Web Worker, image opening |
| `static/features.js`, `filters.js` | JS port of the spectral + texture features (scipy.ndimage semantics) |
| `static/forest.js`, `metrics.js` | random forest (predict exported sklearn trees; train), split / report / class stats |
| `static/stretch.js`, `geotiff.js` | display stretch port; GeoTIFF reader/writer |
| `static/*.test.mjs`, `fixture_util.mjs` | node tests against `tests/fixtures/*.json` |

## Tests (`pytest tests`)
`test_imports_light`, `test_spectral`, `test_texture`, `test_feature_set_order`, `test_random_forest` (regressions for
fixed bugs), `test_web` (API, static build, node tests), `test_web_engine` (fixtures current, GeoTIFF writer, helpers), `test_sample_data` (sample.tif layout, current),
`test_gui_offscreen` (MainWindow in a subprocess, skipped without the desktop deps). `tests/gui_scripts/` holds manual
GUI scripts (not collected).
