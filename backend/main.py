from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import os

# Import our custom modules
from data_loader import DataLoader, create_demo_rasters
from risk_engine import RiskScoringEngine, RiskAssessment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global data_loader, risk_engine
    
    logger.info("Starting HillSafe backend...")
    
    # Initialize risk engine
    risk_engine = RiskScoringEngine()
    logger.info("Risk engine initialized")
    
    # Initialize data loader
    data_dir = os.environ.get("HILLSAFE_DATA_DIR", "./data")
    data_loader = DataLoader(data_dir)
    
    # Try to load real data
    success = data_loader.load_all_data()
    
    if not success:
        logger.warning("Real data not available, creating demo rasters...")
        # Create demo data if real data is not available
        create_demo_rasters(data_dir)
        # Try loading again
        success = data_loader.load_all_data()
        
        if success:
            logger.info("Demo data loaded successfully")
        else:
            logger.error("Failed to load even demo data!")
    
    logger.info("HillSafe backend startup complete")
    
    yield
    
    # Cleanup (if needed)
    logger.info("HillSafe backend shutting down")

app = FastAPI(title="HillSafe Backend - Real Data Edition", lifespan=lifespan)

# Development CORS: allow frontend served from file or localhost during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data loader and risk engine
data_loader: Optional[DataLoader] = None
risk_engine: Optional[RiskScoringEngine] = None


class RiskAssessmentResponse(BaseModel):
    lat: float
    lon: float
    overall_risk_score: float  # 0-1 scale
    risk_level: str
    confidence: float
    factor_scores: Dict[str, float]
    missing_data: List[str]
    reasoning: str
    recommendations: List[str]
    raw_data: Dict[str, Any]  # The underlying analysis data


class SystemInfo(BaseModel):
    data_loaded: bool
    available_data_layers: List[str]
    data_extent: Optional[Dict[str, float]]
    notes: str


@app.on_event("startup")
async def startup_event():
    """Initialize data loader and risk engine on startup."""
    global data_loader, risk_engine
    
    logger.info("Starting HillSafe backend...")
    
    # Initialize risk engine
    risk_engine = RiskScoringEngine()
    logger.info("Risk engine initialized")
    
    # Initialize data loader
    data_dir = os.environ.get("HILLSAFE_DATA_DIR", "./data")
    data_loader = DataLoader(data_dir)
    
    # Try to load real data
    success = data_loader.load_all_data()
    
    if not success:
        logger.warning("Real data not available, creating demo rasters...")
        # Create demo data if real data is not available
        create_demo_rasters(data_dir)
        # Try loading again
        success = data_loader.load_all_data()
        
        if success:
            logger.info("Demo data loaded successfully")
        else:
            logger.error("Failed to load even demo data!")
    
    logger.info("HillSafe backend startup complete")


@app.get("/", response_class=RedirectResponse)
def root():
    """Redirect to the served frontend index."""
    return RedirectResponse(url="/static/index.html")


@app.get("/info", response_model=SystemInfo)
def get_system_info():
    """Get information about loaded data and system status."""
    if not data_loader:
        raise HTTPException(status_code=500, detail="Data loader not initialized")
    
    available_layers = []
    if data_loader.slope_raster:
        available_layers.append("slope")
    if data_loader.elevation_raster:
        available_layers.append("elevation") 
    if data_loader.distance_to_water_raster:
        available_layers.append("water_distance")
    
    # Try to determine data extent from slope raster (our reference)
    extent = None
    if data_loader.slope_raster:
        transform = data_loader.slope_raster.transform
        width = data_loader.slope_raster.width
        height = data_loader.slope_raster.height
        
        # Calculate bounds
        west = transform[2]
        north = transform[5] 
        east = west + width * transform[0]
        south = north + height * transform[4]
        
        extent = {
            "west": west,
            "south": south, 
            "east": east,
            "north": north
        }
    
    return SystemInfo(
        data_loaded=len(available_layers) > 0,
        available_data_layers=available_layers,
        data_extent=extent,
        notes="Real GeoTIFF data loaded" if available_layers else "No spatial data loaded"
    )


@app.get("/check", response_model=RiskAssessmentResponse)
def assess_location_risk(lat: float, lon: float):
    """Perform comprehensive risk assessment for a location."""
    if not data_loader or not risk_engine:
        raise HTTPException(status_code=500, detail="Backend not properly initialized")
    
    # Get analysis data from all available layers
    analysis_data = data_loader.get_analysis_data(lon, lat)
    
    # Check if location is within data extent
    if not analysis_data['data_available']:
        raise HTTPException(
            status_code=400, 
            detail=f"Location ({lat:.6f}, {lon:.6f}) is outside the available data extent"
        )
    
    # Perform risk assessment
    assessment = risk_engine.assess_risk(analysis_data)
    
    # Generate recommendations
    recommendations = risk_engine.get_recommendations(assessment)
    
    return RiskAssessmentResponse(
        lat=lat,
        lon=lon,
        overall_risk_score=assessment.overall_score,
        risk_level=assessment.risk_level,
        confidence=assessment.confidence,
        factor_scores=assessment.factor_scores,
        missing_data=assessment.missing_data,
        reasoning=assessment.reasoning,
        recommendations=recommendations,
        raw_data=analysis_data
    )


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "data_loaded": data_loader is not None and (
            data_loader.slope_raster is not None or 
            data_loader.elevation_raster is not None
        ),
        "risk_engine": risk_engine is not None
    }


# Serve the frontend static files from the `frontend` folder
HERE = os.path.dirname(__file__)
FRONTEND_DIR = os.path.abspath(os.path.join(HERE, "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    logger.info(f"Serving frontend from {FRONTEND_DIR}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)