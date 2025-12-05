# Data Processing 

This folder holds scripts to prepare real DEM, slope, water proximity rasters for Sri Lanka.

## Data Processing with `prepare_data.py`

The main script `prepare_data.py` handles:

1. **Input Data Sources**:
   - **Copernicus DEM GLO-30**: `cop_dem_srilanka.tif` (30m resolution elevation)
   - **GADM Boundaries**: `gadm41_LKA.gpkg` (Administrative districts)
   - **HydroRIVERS**: `water_srilanka.shp` (River network clipped to Sri Lanka)

2. **Processing Capabilities**:
   - Country-wide Sri Lanka processing
   - District-specific processing
   - Automated slope calculation from DEM
   - Water proximity distance computation
   - Multi-district boundary support

3. **Output Products**:
   - `slope_srilanka.tif` - Slope in degrees
   - `water_distance_srilanka.tif` - Distance to nearest water body
   - `districts_srilanka.geojson` - Processed district boundaries

## Usage Examples

```bash
# Process entire Sri Lanka
python prepare_data.py --country-mode \
  --dem cop_dem_srilanka.tif \
  --districts gadm41_LKA.gpkg \
  --water water_srilanka.shp

# Process specific district
python prepare_data.py --district-name "Colombo" \
  --dem cop_dem_srilanka.tif \
  --districts gadm41_LKA.gpkg \
  --water water_srilanka.shp
```
