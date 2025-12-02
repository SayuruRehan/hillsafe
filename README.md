# HillSafe — Housing Suitability Map (Real Data Edition)

This repository contains a **real GeoTIFF-powered** implementation of HillSafe for housing suitability assessment in Sri Lanka's hill country. It uses actual elevation, slope, and water proximity data with a multi-factor risk scoring engine.

## 🚀 Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Application
```bash
./start.sh  # or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the App
Open your browser to: **http://127.0.0.1:8000/**

Click anywhere on the map to get a comprehensive risk assessment with:
- **Overall Risk Score** (0-100%)
- **Risk Level**: Low/Moderate/High/Very High
- **Individual Factor Scores**: Slope, Water Proximity, Elevation
- **Actionable Recommendations**
- **Confidence Level** based on data availability

## 📊 Real Data Integration

### Using Your Own GeoTIFF Data

1. **Prepare your data** using the included script:
```bash
cd data_processing
python prepare_data.py \
  --dem your_dem.tif \
  --water-bodies rivers.geojson \
  --district kandy_boundary.geojson \
  --compute-slope \
  --output-dir ../backend/data
```

2. **Place files in `backend/data/`**:
   - `dem.tif` or `slope.tif` (required)
   - `water_bodies.geojson` (optional)
   - `district_boundary.geojson` (optional)

### Demo Data

If no real data is provided, the system automatically generates realistic demo rasters covering a Kandy-area extent for testing.

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
