"""
Data loader for real GeoTIFF rasters and vector data.
This module handles loading DEM, slope, and vector data for HillSafe analysis.
"""

import os
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from scipy.ndimage import distance_transform_edt
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RasterData:
    """Container for raster data with spatial metadata."""
    
    def __init__(self, data: np.ndarray, transform: rasterio.Affine, crs: str, nodata: Optional[float] = None):
        self.data = data
        self.transform = transform
        self.crs = crs
        self.nodata = nodata
        self.height, self.width = data.shape

    def xy_to_pixel(self, x: float, y: float) -> Tuple[Optional[int], Optional[int]]:
        """Convert lon/lat to pixel row/col coordinates."""
        try:
            col, row = ~self.transform * (x, y)
            col, row = int(round(col)), int(round(row))
            if 0 <= row < self.height and 0 <= col < self.width:
                return row, col
            return None, None
        except Exception:
            return None, None

    def get_value(self, x: float, y: float) -> Optional[float]:
        """Get raster value at lon/lat coordinate."""
        row, col = self.xy_to_pixel(x, y)
        if row is not None and col is not None:
            value = self.data[row, col]
            if self.nodata is not None and np.isclose(value, self.nodata):
                return None
            return float(value)
        return None


class DataLoader:
    """Loads and manages spatial data for HillSafe analysis."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.dem_raster: Optional[RasterData] = None
        self.slope_raster: Optional[RasterData] = None
        self.distance_to_water_raster: Optional[RasterData] = None
        self.elevation_raster: Optional[RasterData] = None
        
        # Default file paths (can be overridden)
        self.file_paths = {
            'dem': os.path.join(data_dir, 'dem.tif'),
            'slope': os.path.join(data_dir, 'slope.tif'),
            'water_bodies': os.path.join(data_dir, 'water_bodies.geojson'),
            'district_boundary': os.path.join(data_dir, 'district_boundary.geojson')
        }

    def load_raster(self, file_path: str, description: str = "") -> Optional[RasterData]:
        """Load a single raster file."""
        if not os.path.exists(file_path):
            logger.warning(f"{description} raster not found: {file_path}")
            return None
        
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)  # Read first band
                transform = src.transform
                crs = src.crs.to_string() if src.crs else "EPSG:4326"
                nodata = src.nodata
                
            logger.info(f"Loaded {description} raster: {data.shape} pixels, CRS: {crs}")
            return RasterData(data, transform, crs, nodata)
            
        except Exception as e:
            logger.error(f"Failed to load {description} raster {file_path}: {e}")
            return None

    def compute_slope_from_dem(self, dem_raster: RasterData) -> Optional[RasterData]:
        """Compute slope in degrees from DEM."""
        try:
            # Get pixel resolution (assuming square pixels in meters)
            pixel_size = abs(dem_raster.transform[0])  # degrees
            # Convert to approximate meters (rough conversion at equator)
            pixel_size_m = pixel_size * 111320  # meters per degree longitude
            
            # Compute gradients
            dy, dx = np.gradient(dem_raster.data.astype(np.float32))
            dy = dy / pixel_size_m
            dx = dx / pixel_size_m
            
            # Calculate slope in degrees
            slope = np.arctan(np.sqrt(dx*dx + dy*dy)) * 180.0 / np.pi
            
            logger.info(f"Computed slope from DEM. Min: {slope.min():.1f}°, Max: {slope.max():.1f}°, Mean: {slope.mean():.1f}°")
            
            return RasterData(slope, dem_raster.transform, dem_raster.crs, dem_raster.nodata)
            
        except Exception as e:
            logger.error(f"Failed to compute slope from DEM: {e}")
            return None

    def rasterize_water_bodies(self, vector_file: str, reference_raster: RasterData, buffer_distance: float = 50.0) -> Optional[RasterData]:
        """Rasterize water bodies and compute distance to nearest water."""
        if not os.path.exists(vector_file):
            logger.warning(f"Water bodies file not found: {vector_file}")
            return None
            
        try:
            # Load water bodies
            water_gdf = gpd.read_file(vector_file)
            
            # Ensure same CRS
            if water_gdf.crs.to_string() != reference_raster.crs:
                water_gdf = water_gdf.to_crs(reference_raster.crs)
            
            # Create raster mask for water bodies
            shapes = [(geom, 1) for geom in water_gdf.geometry]
            water_mask = rasterize(
                shapes,
                out_shape=(reference_raster.height, reference_raster.width),
                transform=reference_raster.transform,
                fill=0,
                dtype=np.uint8
            )
            
            # Compute distance to nearest water in pixels
            distance_pixels = distance_transform_edt(~water_mask.astype(bool))
            
            # Convert to meters (approximate)
            pixel_size_m = abs(reference_raster.transform[0]) * 111320
            distance_meters = distance_pixels * pixel_size_m
            
            logger.info(f"Computed water distance. Max distance: {distance_meters.max():.0f}m")
            
            return RasterData(distance_meters, reference_raster.transform, reference_raster.crs)
            
        except Exception as e:
            logger.error(f"Failed to process water bodies: {e}")
            return None

    def load_all_data(self) -> bool:
        """Load all required spatial data."""
        logger.info("Loading spatial data...")
        
        # Try to load pre-computed slope raster first
        self.slope_raster = self.load_raster(self.file_paths['slope'], "slope")
        
        # If no slope raster, try to load DEM and compute slope
        if self.slope_raster is None:
            self.dem_raster = self.load_raster(self.file_paths['dem'], "DEM")
            if self.dem_raster is not None:
                self.slope_raster = self.compute_slope_from_dem(self.dem_raster)
                self.elevation_raster = self.dem_raster
        
        if self.slope_raster is None:
            logger.error("No slope data available (provide slope.tif or dem.tif)")
            return False
        
        # Use slope raster as reference for water distance computation
        reference_raster = self.slope_raster
        
        # Compute distance to water bodies
        self.distance_to_water_raster = self.rasterize_water_bodies(
            self.file_paths['water_bodies'], 
            reference_raster
        )
        
        # Log what we successfully loaded
        loaded_layers = []
        if self.slope_raster: loaded_layers.append("slope")
        if self.elevation_raster: loaded_layers.append("elevation")
        if self.distance_to_water_raster: loaded_layers.append("water_distance")
        
        logger.info(f"Successfully loaded: {', '.join(loaded_layers)}")
        return len(loaded_layers) > 0

    def get_analysis_data(self, lon: float, lat: float) -> Dict[str, Any]:
        """Get all analysis data for a specific coordinate."""
        result = {
            'lon': lon,
            'lat': lat,
            'slope_deg': None,
            'elevation_m': None,
            'distance_to_water_m': None,
            'data_available': []
        }
        
        if self.slope_raster:
            result['slope_deg'] = self.slope_raster.get_value(lon, lat)
            if result['slope_deg'] is not None:
                result['data_available'].append('slope')
        
        if self.elevation_raster:
            result['elevation_m'] = self.elevation_raster.get_value(lon, lat)
            if result['elevation_m'] is not None:
                result['data_available'].append('elevation')
        
        if self.distance_to_water_raster:
            result['distance_to_water_m'] = self.distance_to_water_raster.get_value(lon, lat)
            if result['distance_to_water_m'] is not None:
                result['data_available'].append('water_distance')
        
        return result


def create_demo_rasters(data_dir: str) -> bool:
    """Create demo GeoTIFF files for testing when real data is not available."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Demo extent (Kandy area)
    west, south, east, north = 80.2, 7.0, 81.0, 7.6
    width, height = 800, 600
    
    # Create transform
    transform = rasterio.transform.from_bounds(west, south, east, north, width, height)
    
    # Create demo DEM (elevation data)
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Synthetic elevation: higher in the center/edges
    elevation = 500 + 300 * (1 - (X**2 + Y**2)) + 50 * np.random.RandomState(42).randn(height, width)
    elevation = np.clip(elevation, 100, 1500).astype(np.float32)
    
    # Write demo DEM
    dem_path = os.path.join(data_dir, 'dem.tif')
    with rasterio.open(
        dem_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=elevation.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(elevation, 1)
    
    # Create demo slope (computed from elevation)
    pixel_size_m = 111320 * abs(transform[0])  # rough conversion
    dy, dx = np.gradient(elevation.astype(np.float32))
    dy = dy / pixel_size_m
    dx = dx / pixel_size_m
    slope = np.arctan(np.sqrt(dx*dx + dy*dy)) * 180.0 / np.pi
    
    slope_path = os.path.join(data_dir, 'slope.tif')
    with rasterio.open(
        slope_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=slope.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(slope, 1)
    
    logger.info(f"Created demo rasters in {data_dir}")
    logger.info(f"DEM range: {elevation.min():.0f} - {elevation.max():.0f} m")
    logger.info(f"Slope range: {slope.min():.1f} - {slope.max():.1f}°")
    
    return True