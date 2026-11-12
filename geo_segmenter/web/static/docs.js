// Method and About pages (content from the repo: README, config.py and the code).
import { app, route } from './core.js';
import { getInfo } from './state.js';

route('method', async () => {
  const info = await getInfo(), f = info.features, d = info.defaults;
  app().innerHTML = `
  <h1>Method</h1>
  <p class="lead">The web pages run the same steps as the desktop app's <i>Train Model</i> and <i>Run Segmentation</i>
    (gui/main_window.py), which use models/pipeline.py, the feature extractors and RandomForestModel.</p>
  <h2>1. Image and band roles</h2>
  <p>The raster is read as (height, width, bands). Bands 1-3 are taken as red, green, blue and band 4, if present, as
    near-infrared (<code>pipeline.default_band_map</code>); the Bands panel lets you change these roles. Display uses
    the desktop Stretch dialog's options (min-max, linear, 2%/5% percentile, ±2 standard deviations).</p>
  <h2>2. Features per pixel</h2>
  <ul>
    <li><b>Spectral</b> (three or more bands): ${f.spectral_indices.join(', ')} where their bands are mapped, then the
      red/green and NIR/red ratios. <span class="eq">NDVI = (NIR − R) / (NIR + R + 1e-8)</span></li>
    <li><b>Texture</b>: the grey image 0.2989 R + 0.5870 G + 0.1140 B is stretched to 0-255; for windows of
      ${f.texture_windows.join(', ')} px: local standard deviation, local range (max − min) and Gaussian gradient
      magnitude with σ = window / 3 (scipy.ndimage, mode 'reflect').</li>
  </ul>
  <h2>3. Training</h2>
  <p>The painted pixels of each class are the training samples. RandomForestModel holds out a stratified 20% for
    validation and fits a scikit-learn random forest: ${d.n_estimators} trees, max depth ${d.max_depth},
    max_features '${d.max_features}', class_weight '${d.class_weight}'. It reports training and validation accuracy,
    a classification report and impurity-based feature importances.</p>
  <h2>4. Classification</h2>
  <p>Every pixel's features go through the forest; each pixel gets the class with the highest mean tree probability.
    Class statistics count pixels per class and convert to hectares when the GeoTIFF gives a metric pixel size.</p>
  <h2>What the browser port reproduces</h2>
  <ul>
    <li>Feature maps: equal to the Python extractors to about 1e-15 relative error (node tests on demo crops, a uint8
      RGB image and a one-band float image).</li>
    <li>Display stretch: identical bytes to <code>utils/display_stretch.prepare_display</code> for every stretch type.</li>
    <li>A forest trained in Python and exported gives sklearn's labels and probabilities exactly in JS; on the demo, JS
      features + those trees reproduce the Python classification pixel for pixel (button under the confusion matrix).</li>
    <li>Training in JS is the same algorithm with different random draws (bootstrap, feature sampling, validation
      split), so a JS forest is a different forest of the same kind: on a demo crop it agrees with the Python labels
      on about 99% of pixels.</li>
    <li>The GeoTIFF reader handles strips and tiles, chunky or planar bands, 8-64-bit integer and float samples,
      LZW / Deflate / PackBits, and reads the georeferencing tags; tested on files written by tifffile.</li>
  </ul>
  <h2>Desktop only</h2>
  <p>The CNN model (PyTorch), terrain features for elevation data, LIDAR point clouds (laspy/PDAL), vector layers,
    base maps and GPS are in the desktop app only. Very large images are block-averaged in the browser to at most 1.5
    million pixels.</p>`;
});

route('about', async () => {
  const info = await getInfo();
  app().innerHTML = `
  <h1>About</h1>
  <p class="lead">${info.title} ${info.version}: a Python desktop application for segmenting and classifying geospatial
    imagery (satellite and aerial images, LIDAR) with machine learning, with this web interface for its classical
    random-forest pipeline.</p>
  <h2>The desktop app</h2>
  <pre>conda install -c conda-forge pdal python-pdal   # PDAL is not pip-installable
pip install PyQt6 rasterio pyproj geopandas scikit-learn scikit-image scipy torch laspy shapely pyserial requests matplotlib
python main.py</pre>
  <p class="small mute">These are the packages the code imports (setup.py lists them too).</p>
  <p>Import a raster (File &gt; Import), open the Label Manager, paint training areas with Shift + left click, train a
    Random Forest or CNN and run the segmentation.</p>
  <h2>The web app</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py                 # http://127.0.0.1:8235</pre>
  <p>The local web app adds a Python engine: training and classification with the repo's own RandomForestModel.
    The static site (GitHub Pages) runs everything in the browser with the JS port.</p>
  <h2>Demo data</h2>
  <p>The demo scene is synthetic: generated with a fixed seed by <code>geo_segmenter/utils/synthetic_scene.py</code>
    (random land-cover layout, generic spectra of water, forest, cropland, bare soil and built-up surfaces, sensor
    blur and noise). It is not a real place and its numbers say nothing about real land cover. The desktop sample raster
    test_data/sample.tif is synthetic too (scripts/make_sample_data.py, same generator, 1001 x 1001 px).</p>
  <h2>Credits</h2>
  <p>By George Dickinson; MIT licence. Built on PyQt6, GDAL/rasterio, NumPy, SciPy, scikit-image, scikit-learn and
    PyTorch; inspired by QGIS and ilastik (README).</p>`;
});
