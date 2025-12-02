# HillSafe — Housing Suitability Map (MVP scaffold)

This repository contains a minimal scaffold for the HillSafe project (see the PRD provided by the owner).

What is included in this scaffold:

- `backend/` - a FastAPI demo backend (`main.py`) that serves `/check` and `/info` using synthetic demo rasters for development.
- `frontend/` - a simple Leaflet-based static page (`index.html`) that calls `/check` and shows results when you click the map.
- `data_processing/` - notes describing the offline processing steps to create real DEM/slope/safe_mask rasters.

Important: This scaffold uses synthetic, generated data for development and demo purposes. Replace with real rasters (DEM, slope, river masks) to run true analyses.

Quick start (development):

1. Create a Python environment and install backend dependencies:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the backend (dev mode):

```bash
./start.sh
```

This starts the backend on `http://0.0.0.0:8000`.

3. Open the demo frontend:

- Either open `frontend/index.html` in a browser (it will call `http://localhost:8000/check` if backend is running on port 8000),
- Or serve the frontend directory with a static server, e.g. `python -m http.server` from the `frontend/` folder and open the page.

Files to replace when moving to real data:
- `backend/main.py` currently uses synthetic arrays. Replace the demo arrays with real rasters (load with `rasterio`) and implement `latlon_to_indices` using the raster transform.

Notes and future work:
- Add a React + Vite frontend if you prefer a React-based UI.
- Add `data_processing/process.ipynb` with runnable steps to produce `slope_kandy.tif` and `safe_mask_kandy.tif`.
- Add tests and a CI workflow.

Disclaimer: This is an educational demo. It is NOT an official hazard product and does not replace engineering assessments.
