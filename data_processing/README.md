# Data Processing (notes)

This folder holds scripts and notebooks used to prepare real DEM, slope, river rasters and the `safe_mask`.

Steps to produce real rasters (offline):

1. Obtain data:
   - DEM (SRTM 30m or similar) covering pilot district (e.g., Kandy).
   - District boundary polygon (GeoJSON / shapefile).
   - Rivers/streams (OpenStreetMap exports or HydroSHEDS).

2. Clip DEM to district boundary:
   - Use `rasterio.mask` or `gdalwarp` to clip DEM using the district polygon.

3. Compute slope from DEM:
   - Rasterio + numpy: compute gradient and convert to degrees.
   - Or use `gdaldem slope`.

4. Rasterize rivers to the DEM grid:
   - Use `rasterio.features.rasterize` with the same transform and shape as the DEM.

5. Compute distance to nearest river:
   - Use `scipy.ndimage.distance_transform_edt` on the inverted river mask, then multiply by pixel resolution (meters).

6. Create `safe_mask`:
   - safe = (slope_deg <= SLOPE_THRESHOLD) & (distance_to_river >= RIVER_BUFFER)

7. Export outputs:
   - `slope_kandy.tif`, `distance_to_river_kandy.tif`, `safe_mask_kandy.tif`

This folder can later include an executable notebook `process.ipynb` with code snippets.
