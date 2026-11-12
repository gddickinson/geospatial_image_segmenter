"""Web interface (FastAPI + plain ES-module front end) and static GitHub Pages site.

    app.py               FastAPI app: /api/info, /api/demo (GET, exported to the static site), /api/classify (POST)
    classify_service.py  runs models/pipeline.py + RandomForestModel; base64 / mask packing; forest export for JS
    demo.py              synthetic demo scene classified by the Python pipeline (cached)
    static/              front end: pages, JS ports (features, filters, forest, stretch, geotiff) and node tests
"""
