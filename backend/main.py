from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import Dict

app = FastAPI(title="HillSafe Backend (demo)")

# Development CORS: allow frontend served from file or localhost during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo spatial grid covering a small area near Kandy (approx)
MIN_LAT = 7.0
MAX_LAT = 7.6
MIN_LON = 80.2
MAX_LON = 81.0
ROWS = 600  # ~0.001 deg ~111m per pixel on lat -> ~60km / 600 -> ~100m per pixel rough
COLS = 800

SLOPE_THRESHOLD_DEG = 30.0
RIVER_BUFFER_M = 50.0

# Prepare demo slope raster and a simple river mask
latitudes = np.linspace(MAX_LAT, MIN_LAT, ROWS)  # top->bottom
longitudes = np.linspace(MIN_LON, MAX_LON, COLS)
lat_res = (MAX_LAT - MIN_LAT) / ROWS
lon_res = (MAX_LON - MIN_LON) / COLS

# Create a slope field: low slopes near valley center, steeper on sides
X = np.linspace(-1, 1, COLS)
Y = np.linspace(-1, 1, ROWS)[:, None]
slope_field = (np.abs(X) + np.abs(Y)) * 20 + np.random.RandomState(0).randn(ROWS, COLS) * 1.5
slope_field = np.clip(slope_field, 0, 60)  # degrees

# Define a synthetic river as a vertical band near the center longitude
river_lon = (MIN_LON + MAX_LON) / 2
river_col = int((river_lon - MIN_LON) / (MAX_LON - MIN_LON) * COLS)
river_mask = np.zeros((ROWS, COLS), dtype=np.uint8)
river_width_pixels = max(1, int( (50.0 / 111320) / lon_res ))  # ~50m in lon degrees -> pixels
river_mask[:, max(0, river_col - river_width_pixels):min(COLS, river_col + river_width_pixels+1)] = 1

# Precompute distance_to_river in meters using approximate conversion at mean latitude
mean_lat = 0.5 * (MIN_LAT + MAX_LAT)
meters_per_deg_lat = 111132.0
meters_per_deg_lon = 111320.0 * abs(np.cos(np.deg2rad(mean_lat)))

# compute lon of each column, compute distance in meters to river_lon
col_lons = longitudes
dist_lon_m = np.abs(col_lons - river_lon) * meters_per_deg_lon
# For simplicity, take distance as horizontal distance only (ignoring lat diff)
distance_to_river = np.tile(dist_lon_m[None, :], (ROWS, 1))

# safe mask (1 safe, 0 unsafe): slope <= threshold and distance >= buffer
safe_mask = np.where((slope_field <= SLOPE_THRESHOLD_DEG) & (distance_to_river >= RIVER_BUFFER_M), 1, 0)

class CheckResponse(BaseModel):
    lat: float
    lon: float
    slope_deg: float
    is_safe_basic: bool
    rules: Dict[str, float]
    reason: str


def latlon_to_indices(lat: float, lon: float):
    if not (MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON):
        return None
    # row index: lat decreases as row increases (we used MAX_LAT -> MIN_LAT)
    row = int((MAX_LAT - lat) / (MAX_LAT - MIN_LAT) * ROWS)
    col = int((lon - MIN_LON) / (MAX_LON - MIN_LON) * COLS)
    # clamp
    row = min(max(row, 0), ROWS - 1)
    col = min(max(col, 0), COLS - 1)
    return row, col


@app.get("/check", response_model=CheckResponse)
def check(lat: float, lon: float):
    coords = latlon_to_indices(lat, lon)
    if coords is None:
        raise HTTPException(status_code=400, detail="Coordinates are outside the demo extent.")
    r, c = coords
    slope = float(round(float(slope_field[r, c]), 2))
    dist = float(round(float(distance_to_river[r, c]), 1))
    is_safe = (slope <= SLOPE_THRESHOLD_DEG) and (dist >= RIVER_BUFFER_M)
    reason_parts = []
    reason_parts.append(f"Slope {slope}° {'<=' if slope <= SLOPE_THRESHOLD_DEG else '>'} {SLOPE_THRESHOLD_DEG}°")
    reason_parts.append(f"distance_to_river {int(dist)}m {'>=' if dist >= RIVER_BUFFER_M else '<'} {int(RIVER_BUFFER_M)}m")
    reason = ", ".join(reason_parts)
    return CheckResponse(
        lat=lat,
        lon=lon,
        slope_deg=slope,
        is_safe_basic=bool(is_safe),
        rules={
            "slope_threshold_deg": SLOPE_THRESHOLD_DEG,
            "river_buffer_m": RIVER_BUFFER_M,
        },
        reason=reason,
    )


@app.get("/")
def root():
    # Redirect to the served frontend index (served from /static)
    return RedirectResponse(url="/static/index.html")


# Serve the frontend static files from the `frontend` folder so the app can be
# opened from the backend origin (avoids CORS and cross-origin fetch issues
# during local development).
import os
HERE = os.path.dirname(__file__)
FRONTEND_DIR = os.path.abspath(os.path.join(HERE, "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/info")
def info():
    return {
        "demo_extent": {
            "min_lat": MIN_LAT,
            "max_lat": MAX_LAT,
            "min_lon": MIN_LON,
            "max_lon": MAX_LON,
        },
        "rows": ROWS,
        "cols": COLS,
        "notes": "This is a demo backend using synthetic slope and river data. Replace with real rasters for production.",
    }
