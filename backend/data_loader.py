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
from typing import Tuple, Optional, Dict, Any, List
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
        self.districts_gdf: Optional[gpd.GeoDataFrame] = None
        
        # File patterns for different data modes
        self.file_patterns = {
            # Country-wide files
            'dem_country': os.path.join(data_dir, 'dem_srilanka.tif'),
            'slope_country': os.path.join(data_dir, 'slope_srilanka.tif'),
            'water_distance_country': os.path.join(data_dir, 'water_distance_srilanka.tif'),
            'districts': os.path.join(data_dir, 'districts_srilanka.geojson'),
            
            # Legacy single files
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
        
        # First, try to load Sri Lanka country-wide data
        if self._load_country_data():
            logger.info("Successfully loaded Sri Lanka country-wide data")
            return True
        
        # Fallback to legacy single-file approach
        return self._load_legacy_data()

    def _load_country_data(self) -> bool:
        """Load Sri Lanka country-wide data."""
        
        # Load districts
        if os.path.exists(self.file_patterns['districts']):
            try:
                self.districts_gdf = gpd.read_file(self.file_patterns['districts'])
                logger.info(f"Loaded {len(self.districts_gdf)} districts")
            except Exception as e:
                logger.warning(f"Could not load districts: {e}")
        
        # Try to load country-wide slope raster
        self.slope_raster = self.load_raster(self.file_patterns['slope_country'], "slope (country)")
        
        # Try to load country-wide DEM
        if not self.slope_raster:
            self.dem_raster = self.load_raster(self.file_patterns['dem_country'], "DEM (country)")
            if self.dem_raster:
                self.slope_raster = self.compute_slope_from_dem(self.dem_raster)
                self.elevation_raster = self.dem_raster
        
        # Try to load pre-computed water distance
        self.distance_to_water_raster = self.load_raster(self.file_patterns['water_distance_country'], 
                                                        "water distance (country)")
        
        # Check if we have at least slope data
        if self.slope_raster is None:
            return False
        
        # Log what we successfully loaded
        loaded_layers = []
        if self.slope_raster: loaded_layers.append("slope")
        if self.elevation_raster: loaded_layers.append("elevation")
        if self.distance_to_water_raster: loaded_layers.append("water_distance")
        if self.districts_gdf is not None: loaded_layers.append("districts")
        
        logger.info(f"Successfully loaded country data: {', '.join(loaded_layers)}")
        return len(loaded_layers) > 0

    def _load_legacy_data(self) -> bool:
        """Load data using legacy single-file approach."""
        logger.info("Trying legacy data loading...")
        
        # Try to load pre-computed slope raster first
        self.slope_raster = self.load_raster(self.file_patterns['slope'], "slope")
        
        # If no slope raster, try to load DEM and compute slope
        if self.slope_raster is None:
            self.dem_raster = self.load_raster(self.file_patterns['dem'], "DEM")
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
            self.file_patterns['water_bodies'], 
            reference_raster
        )
        
        # Log what we successfully loaded
        loaded_layers = []
        if self.slope_raster: loaded_layers.append("slope")
        if self.elevation_raster: loaded_layers.append("elevation")
        if self.distance_to_water_raster: loaded_layers.append("water_distance")
        
        logger.info(f"Successfully loaded legacy data: {', '.join(loaded_layers)}")
        return len(loaded_layers) > 0

    def get_analysis_data(self, lon: float, lat: float) -> Dict[str, Any]:
        """Get all analysis data for a specific coordinate."""
        result = {
            'lon': lon,
            'lat': lat,
            'slope_deg': None,
            'elevation_m': None,
            'distance_to_water_m': None,
            'district_name': None,
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
        
        # Get district information if available
        if self.districts_gdf is not None:
            try:
                from shapely.geometry import Point
                point = Point(lon, lat)
                # Find which district contains this point
                containing_districts = self.districts_gdf[self.districts_gdf.geometry.contains(point)]
                if not containing_districts.empty:
                    result['district_name'] = containing_districts.iloc[0]['NAME_2']
                    result['data_available'].append('district')
            except Exception as e:
                logger.debug(f"Could not determine district for {lat}, {lon}: {e}")
        
        return result

    def get_districts_list(self) -> List[Dict[str, Any]]:
        """Get list of available districts with their bounds."""
        if self.districts_gdf is None:
            return []
        
        districts = []
        for _, row in self.districts_gdf.iterrows():
            bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
            districts.append({
                'name': row['NAME_2'],
                'bounds': {
                    'west': bounds[0],
                    'south': bounds[1], 
                    'east': bounds[2],
                    'north': bounds[3]
                }
            })
        
        return sorted(districts, key=lambda x: x['name'])

    def get_data_extent(self) -> Optional[Dict[str, float]]:
        """Get the extent of loaded data."""
        if self.slope_raster:
            transform = self.slope_raster.transform
            width = self.slope_raster.width
            height = self.slope_raster.height
            
            # Calculate bounds
            west = transform[2]
            north = transform[5] 
            east = west + width * transform[0]
            south = north + height * transform[4]
            
            return {
                "west": west,
                "south": south, 
                "east": east,
                "north": north
            }
        return None


def create_demo_rasters(data_dir: str) -> bool:
    """Create demo GeoTIFF files for testing when real data is not available."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Sri Lanka extent (approximate)
    west, south, east, north = 79.5, 5.9, 81.9, 9.9
    width, height = 2400, 4000  # Higher resolution for country-wide
    
    # Create transform
    transform = rasterio.transform.from_bounds(west, south, east, north, width, height)
    
    # Create demo DEM (elevation data) - more realistic for Sri Lanka
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Synthetic elevation: mountainous center, coastal plains
    # Central highlands (around 0.3-0.7 in both x,y)
    center_x, center_y = 0.5, 0.6  # Roughly central Sri Lanka
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Create elevation pattern
    elevation = (
        # Base sea level to low hills
        50 + 200 * (1 - Y) +  # Elevation increases going north
        # Central highlands
        1000 * np.exp(-dist_from_center * 8) +
        # Add some noise for realism
        100 * np.random.RandomState(42).randn(height, width)
    )
    
    elevation = np.clip(elevation, 0, 2500).astype(np.float32)
    
    # Write demo DEM - use Sri Lanka naming
    dem_path = os.path.join(data_dir, 'dem_srilanka.tif')
    with rasterio.open(
        dem_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=elevation.dtype,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(elevation, 1)
    
    # Create demo slope (computed from elevation)
    pixel_size_m = 111320 * abs(transform[0])  # rough conversion
    dy, dx = np.gradient(elevation.astype(np.float32))
    dy = dy / pixel_size_m
    dx = dx / pixel_size_m
    slope = np.arctan(np.sqrt(dx*dx + dy*dy)) * 180.0 / np.pi
    
    slope_path = os.path.join(data_dir, 'slope_srilanka.tif')
    with rasterio.open(
        slope_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=slope.dtype,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(slope, 1)
    
    # Create demo water distance (distance from major rivers)
    water_distance = np.ones((height, width), dtype=np.float32) * 5000  # Start with 5km
    
    # Add some "rivers" as lines of low distance
    for river_start in [(0.2, 0.8), (0.5, 0.9), (0.7, 0.7)]:
        for i in range(height):
            for j in range(width):
                y_norm = i / height
                x_norm = j / width
                # Distance to river line
                river_dist = abs(x_norm - (river_start[0] + (river_start[1] - river_start[0]) * y_norm))
                river_dist_m = river_dist * 111320 * abs(transform[0]) * 50  # Scale to meters
                water_distance[i, j] = min(water_distance[i, j], river_dist_m)
    
    water_dist_path = os.path.join(data_dir, 'water_distance_srilanka.tif')
    with rasterio.open(
        water_dist_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=water_distance.dtype,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(water_distance, 1)
    
    # Create demo districts data
    districts_data = {
        'NAME_2': ['Colombo', 'Kandy', 'Galle', 'Jaffna', 'Anuradhapura', 'Batticaloa'],
        'geometry': []
    }
    
    # Create simple rectangular districts
    for i, name in enumerate(districts_data['NAME_2']):
        # Divide the country into rough districts
        col = i % 2
        row = i // 2
        
        district_west = west + col * (east - west) / 2
        district_east = west + (col + 1) * (east - west) / 2
        district_south = south + row * (north - south) / 3
        district_north = south + (row + 1) * (north - south) / 3
        
        from shapely.geometry import box
        districts_data['geometry'].append(
            box(district_west, district_south, district_east, district_north)
        )
    
    # Save demo districts
    import geopandas as gpd
    districts_gdf = gpd.GeoDataFrame(districts_data, crs='EPSG:4326')
    districts_path = os.path.join(data_dir, 'districts_srilanka.geojson')
    districts_gdf.to_file(districts_path, driver="GeoJSON")
    
    logger.info(f"Created demo rasters for Sri Lanka in {data_dir}")
    logger.info(f"DEM range: {elevation.min():.0f} - {elevation.max():.0f} m")
    logger.info(f"Slope range: {slope.min():.1f} - {slope.max():.1f}°")
    logger.info(f"Water distance range: {water_distance.min():.0f} - {water_distance.max():.0f} m")
    logger.info(f"Created {len(districts_data['NAME_2'])} demo districts")
    
    return True