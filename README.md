# HillSafe — Sri Lanka Housing Suitability Assessment

**Real GeoTIFF-powered** housing suitability assessment for **entire Sri Lanka** using elevation, slope, and water proximity data with multi-factor risk scoring.

## 🇱🇰 Sri Lanka Countrywide Coverage

This system now supports **full Sri Lankan coverage** using:
- **Copernicus DEM GLO-30**: 30m resolution elevation data  
- **GADM Administrative Boundaries**: All 25 districts
- **HydroRIVERS**: Comprehensive river network

## 🚀 Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Preparation (with your real data)

Place your Sri Lankan datasets in `data_processing/`:
```bash
cd data_processing

# For entire Sri Lanka
python prepare_data.py \
  --country-mode \
  --dem cop_dem_srilanka.tif \
  --districts gadm41_LKA.gpkg \
  --water water_srilanka.shp \
  --output-dir ../backend/data

# OR for specific district (e.g., Kandy)
python prepare_data.py \
  --district-name "Kandy" \
  --dem cop_dem_srilanka.tif \
  --districts gadm41_LKA.gpkg \
  --water water_srilanka.shp \
  --output-dir ../backend/data
```

### 3. Start the Application
```bash
cd ../backend
./start.sh
```

### 4. Access the App
Open: **http://127.0.0.1:8000/**

Click anywhere on Sri Lanka to get:
- **Overall Risk Score** (0-100%)
- **Risk Level**: Low/Moderate/High/Very High  
- **District Information**: Automatically detected
- **Individual Factor Analysis**: Slope, Water Proximity, Elevation
- **Actionable Recommendations** with confidence levels

## 📊 Required Sri Lankan Datasets

### Copernicus DEM GLO-30 (`cop_dem_srilanka.tif`)
- **Source**: https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model
- **Resolution**: 30m (1 arc-second)
- **Coverage**: Global, download Sri Lanka tile
- **Format**: GeoTIFF

### GADM Administrative Boundaries (`gadm41_LKA.gpkg`) 
- **Source**: https://gadm.org/download_country.html (Sri Lanka)
- **Levels**: Province, District, DS Division
- **Format**: GeoPackage (.gpkg)
- **Contains**: All 25 districts with proper names

### HydroRIVERS (`water_srilanka.shp`)
- **Source**: https://www.hydrosheds.org/products/hydrorivers  
- **Scale**: 1:50,000 to 1:1,000,000
- **Preprocessing**: Clip Asia dataset to Sri Lanka bounds
- **Format**: Shapefile

#### Quick GeoPandas script to clip HydroRIVERS:
```python
import geopandas as gpd

# Load Asia rivers and Sri Lanka boundary
rivers = gpd.read_file("HydroRIVERS_v10_as.shp")
srilanka = gpd.read_file("gadm41_LKA.gpkg", layer="ADM_0")

# Clip and save
rivers_sl = gpd.clip(rivers, srilanka)
rivers_sl.to_file("water_srilanka.shp")
```

## 🧠 Risk Scoring Engine

The system uses a **multi-factor risk assessment** instead of simple safe/unsafe classification:

### Risk Factors

1. **Terrain Slope** (50% weight)
   - 0-10°: Very Safe (0% risk)
   - 10-15°: Safe (10% risk)  
   - 15-25°: Moderate (30-50% risk)
   - 25-35°: High (50-80% risk)
   - >35°: Very High (80-100% risk)

2. **Water Proximity** (30% weight)
   - >100m: Safe (0% risk)
   - 50-100m: Moderate (20% risk)
   - 20-50m: High (50% risk)
   - <20m: Very High (80% risk)

3. **Elevation** (20% weight)
   - <100m: Drainage issues (30% risk)
   - >2000m: High altitude challenges (20% risk)
   - 100-2000m: Normal range (0% risk)

### Risk Levels
- **Low Risk** (0-25%): ✅ Suitable with standard practices
- **Moderate Risk** (25-50%): ⚠️ Engineering assessment recommended  
- **High Risk** (50-75%): ❌ Extensive mitigation required
- **Very High Risk** (75-100%): 🚫 Alternative site recommended

## 🛠️ Architecture

### Backend (`/backend`)
- **FastAPI** application with real GeoTIFF loading via `rasterio`
- **Data Loader** (`data_loader.py`): Handles DEM, slope, water proximity rasters
- **Risk Engine** (`risk_engine.py`): Multi-factor risk scoring with confidence levels
- **API Endpoints**:
  - `GET /check?lat=...&lon=...` → Risk assessment
  - `GET /info` → Data availability and extent
  - `GET /health` → System status

### Frontend (`/frontend`) 
- **Leaflet-based** interactive map
- Rich popup displays with risk scores, recommendations, and data confidence
- Color-coded risk levels and factor breakdowns

### Data Processing (`/data_processing`)
- **`prepare_data.py`**: Script to clip and process real GeoTIFF data
- **`README.md`**: Notes on offline data preparation steps

## 📝 API Response Example

```json
{
  "lat": 7.3,
  "lon": 80.6,
  "overall_risk_score": 0.35,
  "risk_level": "Moderate Risk",
  "confidence": 0.67,
  "factor_scores": {
    "slope": 0.4,
    "water_proximity": 0.2
  },
  "missing_data": ["Elevation"],
  "reasoning": "Terrain Slope: Moderate Risk (22.3°), Water Proximity: Low Risk (75m from water). Missing: Elevation",
  "recommendations": [
    "⚠️ Suitable with engineering assessment and mitigation measures",
    "• Consider terracing or retaining walls for slope stability"
  ],
  "raw_data": {
    "slope_deg": 22.3,
    "distance_to_water_m": 75.0,
    "elevation_m": null
  }
}
```

## 🗺️ Data Sources (for Real Implementation)

For production use with real data, consider these sources:

### Elevation Data
- **SRTM 30m DEM**: https://earthexplorer.usgs.gov/
- **ALOS PALSAR DEM**: https://search.asf.alaska.edu/
- **Survey Department of Sri Lanka**: Official topographic maps

### Water Bodies
- **OpenStreetMap**: Export waterways for your area
- **HydroSHEDS**: Global drainage database
- **Survey Department**: Official water body maps

### Administrative Boundaries  
- **GADM**: Global administrative boundaries
- **Survey Department**: Official district boundaries

## ⚠️ Important Disclaimers

- **Educational/Research Tool**: Not for official land-use planning decisions
- **Not NBRO Certified**: Does not replace professional geotechnical assessments
- **Local Validation Required**: Ground-truth with local geological knowledge
- **Resolution Limitations**: 30m DEM may miss micro-scale hazards

## 🚧 Development & Extensions

### Adding New Risk Factors
Modify `backend/risk_engine.py` to add factors like:
- Road accessibility
- Land use compatibility  
- Geological hazard zones
- Rainfall/climate data

### Multi-District Support
Update `data_loader.py` to handle multiple district datasets and add district selection to the frontend.

### Machine Learning Integration
The risk engine can be extended with ML-based landslide susceptibility models using the existing data layers as features.

## 📞 Support & Contribution

For issues, improvements, or adding real datasets for Sri Lankan districts, please open an issue or contribute to the repository.

---

**Built with**: FastAPI, Rasterio, GeoPandas, Leaflet, and ❤️ for safer housing in hill country.
