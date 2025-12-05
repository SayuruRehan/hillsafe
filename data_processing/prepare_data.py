#!/usr/bin/env python3
"""
Data preparation script for HillSafe - Sri Lanka Countrywide Edition.
This script processes national datasets for the entire Sri Lanka.

Example usage:
    # Process entire country
    python prepare_data.py --country-mode --dem cop_dem_srilanka.tif --districts gadm41_LKA.gpkg --water water_srilanka.shp
    
    # Process specific district
    python prepare_data.py --district-name "Colombo" --dem cop_dem_srilanka.tif --districts gadm41_LKA.gpkg --water water_srilanka.shp
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import warnings

import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


def load_sri_lanka_districts(districts_path: str):
    """Load Sri Lankan district boundaries from GADM data."""
    logger.info(f"Loading district boundaries from {districts_path}")
    
    # GADM typically has multiple admin levels - we want districts (ADM_ADM_2)
    try:
        districts_gdf = gpd.read_file(districts_path, layer="ADM_ADM_2")
    except:
        # Fallback to other possible layer names
        try:
            districts_gdf = gpd.read_file(districts_path, layer="ADM_2")
        except:
            districts_gdf = gpd.read_file(districts_path)  # Use default layer
    
    # Filter for Sri Lanka if multiple countries present
    if 'GID_0' in districts_gdf.columns:
        districts_gdf = districts_gdf[districts_gdf['GID_0'] == 'LKA']
    elif 'COUNTRY' in districts_gdf.columns:
        districts_gdf = districts_gdf[districts_gdf['COUNTRY'] == 'Sri Lanka']
    
    # Ensure we have the right columns
    if 'NAME_2' not in districts_gdf.columns:
        # Try alternative column names
        name_cols = [col for col in districts_gdf.columns if 'NAME' in col.upper()]
        if name_cols:
            districts_gdf['NAME_2'] = districts_gdf[name_cols[-1]]  # Use the most detailed name
        else:
            logger.error("Could not find district name column in GADM data")
            return None
    
    logger.info(f"Loaded {len(districts_gdf)} districts: {', '.join(districts_gdf['NAME_2'].tolist())}")
    return districts_gdf


def get_district_by_name(districts_gdf: gpd.GeoDataFrame, district_name: str):
    """Get specific district geometry by name."""
    matches = districts_gdf[districts_gdf['NAME_2'].str.contains(district_name, case=False, na=False)]
    
    if len(matches) == 0:
        logger.error(f"District '{district_name}' not found. Available districts:")
        for name in sorted(districts_gdf['NAME_2'].tolist()):
            logger.error(f"  - {name}")
        return None
    elif len(matches) > 1:
        logger.warning(f"Multiple districts match '{district_name}': {matches['NAME_2'].tolist()}")
        return matches.iloc[0:1]  # Return first match
    
    logger.info(f"Found district: {matches.iloc[0]['NAME_2']}")
    return matches


def clip_raster_to_boundary(input_path: str, boundary_gdf: gpd.GeoDataFrame, output_path: str, 
                           target_resolution: float = None):
    """Clip raster to boundary with optional resampling."""
    logger.info(f"Clipping raster {input_path} to boundary")
    
    with rasterio.open(input_path) as src:
        # Ensure same CRS
        if boundary_gdf.crs != src.crs:
            boundary_gdf = boundary_gdf.to_crs(src.crs)
        
        # Clip raster to boundary
        out_image, out_transform = mask(src, boundary_gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        
        # Handle resampling if target resolution specified
        if target_resolution and abs(src.transform[0]) != target_resolution:
            logger.info(f"Resampling from {abs(src.transform[0]):.6f}° to {target_resolution:.6f}°")
            
            # Calculate new dimensions
            bounds = boundary_gdf.total_bounds
            width = int((bounds[2] - bounds[0]) / target_resolution)
            height = int((bounds[3] - bounds[1]) / target_resolution)
            
            # Create new transform
            new_transform = rasterio.transform.from_bounds(
                bounds[0], bounds[1], bounds[2], bounds[3], width, height
            )
            
            # Prepare destination array
            resampled = np.empty((1, height, width), dtype=out_image.dtype)
            
            # Reproject
            reproject(
                source=out_image,
                destination=resampled,
                src_transform=out_transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear
            )
            
            out_image = resampled
            out_transform = new_transform
        
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw"  # Compress to save space
        })
        
        # Write clipped raster
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
    
    # Log statistics
    valid_data = out_image[0][out_image[0] != src.nodata] if src.nodata else out_image[0]
    logger.info(f"Clipped raster saved to {output_path}")
    logger.info(f"  Size: {out_image.shape[2]} x {out_image.shape[1]} pixels")
    logger.info(f"  Value range: {valid_data.min():.1f} - {valid_data.max():.1f}")


def compute_slope_from_dem(dem_path: str, output_path: str):
    """Compute slope in degrees from DEM."""
    logger.info(f"Computing slope from {dem_path}")
    
    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype(np.float32)
        transform = src.transform
        profile = src.profile.copy()
        
        # Handle nodata
        if src.nodata is not None:
            elevation[elevation == src.nodata] = np.nan
        
        # Get pixel resolution in meters
        # For lat/lon data, convert degrees to meters (approximate)
        if src.crs.to_string().startswith('EPSG:4326') or 'WGS 84' in src.crs.to_string():
            # Convert degrees to meters (rough approximation)
            lat = (src.bounds.bottom + src.bounds.top) / 2
            pixel_size_x = abs(transform[0]) * 111320 * np.cos(np.radians(lat))  # lon to meters
            pixel_size_y = abs(transform[4]) * 111320  # lat to meters
        else:
            # Assume CRS is already in meters
            pixel_size_x = abs(transform[0])
            pixel_size_y = abs(transform[4])
        
        # Compute gradients
        dy, dx = np.gradient(elevation)
        dy = dy / pixel_size_y
        dx = dx / pixel_size_x
        
        # Calculate slope in degrees
        slope = np.arctan(np.sqrt(dx*dx + dy*dy)) * 180.0 / np.pi
        
        # Restore nodata areas
        if src.nodata is not None:
            slope[np.isnan(elevation)] = src.nodata
        
        # Update profile for slope output
        profile.update(dtype=slope.dtype, compress='lzw')
        
        # Write slope
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(slope, 1)
    
    valid_slope = slope[~np.isnan(slope)] if src.nodata is not None else slope
    logger.info(f"Slope raster saved to {output_path}")
    logger.info(f"Slope range: {valid_slope.min():.1f}° to {valid_slope.max():.1f}°")


def prepare_water_distance_raster(water_path: str, reference_raster_path: str, output_path: str, 
                                 boundary_gdf: gpd.GeoDataFrame = None):
    """Create distance-to-water raster from vector water bodies."""
    logger.info(f"Processing water bodies from {water_path}")
    
    # Load water bodies
    water_gdf = gpd.read_file(water_path)
    
    # Clip to boundary if provided
    if boundary_gdf is not None:
        if water_gdf.crs != boundary_gdf.crs:
            water_gdf = water_gdf.to_crs(boundary_gdf.crs)
        water_gdf = gpd.clip(water_gdf, boundary_gdf)
    
    # Load reference raster for spatial properties
    with rasterio.open(reference_raster_path) as ref_src:
        ref_transform = ref_src.transform
        ref_shape = (ref_src.height, ref_src.width)
        ref_crs = ref_src.crs
        ref_profile = ref_src.profile.copy()
        
        # Ensure water data has same CRS
        if water_gdf.crs != ref_crs:
            water_gdf = water_gdf.to_crs(ref_crs)
        
        # Rasterize water bodies
        from rasterio.features import rasterize
        shapes = [(geom, 1) for geom in water_gdf.geometry if geom.is_valid]
        
        if not shapes:
            logger.warning("No valid water geometries found")
            # Create empty distance raster
            distance_raster = np.full(ref_shape, 9999.0, dtype=np.float32)
        else:
            water_mask = rasterize(
                shapes,
                out_shape=ref_shape,
                transform=ref_transform,
                fill=0,
                dtype=np.uint8
            )
            
            # Compute distance to nearest water in pixels
            distance_pixels = distance_transform_edt(~water_mask.astype(bool))
            
            # Convert to meters
            if ref_crs.to_string().startswith('EPSG:4326') or 'WGS 84' in ref_crs.to_string():
                # Convert degrees to meters
                bounds = ref_src.bounds
                lat = (bounds.bottom + bounds.top) / 2
                pixel_size_m = abs(ref_transform[0]) * 111320 * np.cos(np.radians(lat))
            else:
                pixel_size_m = abs(ref_transform[0])
            
            distance_raster = distance_pixels * pixel_size_m
        
        # Update profile and write
        ref_profile.update(dtype=distance_raster.dtype, compress='lzw')
        
        with rasterio.open(output_path, 'w', **ref_profile) as dst:
            dst.write(distance_raster, 1)
    
    logger.info(f"Water distance raster saved to {output_path}")
    if distance_raster.max() < 9999:
        logger.info(f"Distance range: 0 - {distance_raster.max():.0f} meters")


def process_country_data(dem_path: str, districts_path: str, water_path: str, output_dir: Path, 
                        target_resolution: float = 0.0008333):  # ~30m at equator
    """Process data for entire Sri Lanka."""
    logger.info("Processing Sri Lanka countrywide data...")
    
    # Load district boundaries
    districts_gdf = load_sri_lanka_districts(districts_path)
    if districts_gdf is None:
        return False
    
    # Get country boundary (union of all districts)
    country_boundary = districts_gdf.dissolve()
    
    # Process DEM
    dem_output = output_dir / "dem_srilanka.tif"
    clip_raster_to_boundary(dem_path, country_boundary, str(dem_output), target_resolution)
    
    # Compute slope
    slope_output = output_dir / "slope_srilanka.tif"
    compute_slope_from_dem(str(dem_output), str(slope_output))
    
    # Process water distance
    water_dist_output = output_dir / "water_distance_srilanka.tif"
    prepare_water_distance_raster(water_path, str(dem_output), str(water_dist_output), country_boundary)
    
    # Save district boundaries
    districts_output = output_dir / "districts_srilanka.geojson"
    districts_gdf.to_file(str(districts_output), driver="GeoJSON")
    
    logger.info(f"Country data processing complete! Files saved to {output_dir}")
    return True


def process_district_data(district_name: str, dem_path: str, districts_path: str, water_path: str, output_dir: Path):
    """Process data for a specific district."""
    logger.info(f"Processing data for {district_name} district...")
    
    # Load district boundaries
    districts_gdf = load_sri_lanka_districts(districts_path)
    if districts_gdf is None:
        return False
    
    # Get specific district
    district_gdf = get_district_by_name(districts_gdf, district_name)
    if district_gdf is None:
        return False
    
    # Clean district name for filenames
    clean_name = district_name.lower().replace(' ', '_')
    
    # Process DEM
    dem_output = output_dir / f"dem_{clean_name}.tif"
    clip_raster_to_boundary(dem_path, district_gdf, str(dem_output))
    
    # Compute slope
    slope_output = output_dir / f"slope_{clean_name}.tif"
    compute_slope_from_dem(str(dem_output), str(slope_output))
    
    # Process water distance
    water_dist_output = output_dir / f"water_distance_{clean_name}.tif"
    prepare_water_distance_raster(water_path, str(dem_output), str(water_dist_output), district_gdf)
    
    # Save district boundary
    district_output = output_dir / f"district_{clean_name}.geojson"
    district_gdf.to_file(str(district_output), driver="GeoJSON")
    
    logger.info(f"District data processing complete! Files saved to {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare spatial data for HillSafe - Sri Lanka Edition")
    parser.add_argument("--dem", required=True, help="Path to Sri Lanka DEM GeoTIFF (e.g., cop_dem_srilanka.tif)")
    parser.add_argument("--districts", required=True, help="Path to GADM districts file (e.g., gadm41_LKA.gpkg)")
    parser.add_argument("--water", required=True, help="Path to water bodies shapefile (e.g., water_srilanka.shp)")
    parser.add_argument("--output-dir", default="../backend/data", help="Output directory for processed files")
    
    # Processing mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--country-mode", action="store_true", help="Process entire Sri Lanka")
    mode_group.add_argument("--district-name", help="Process specific district (e.g., 'Colombo')")
    
    # Optional parameters
    parser.add_argument("--resolution", type=float, default=0.0008333, 
                       help="Target resolution in degrees (~30m at equator)")
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in [args.dem, args.districts, args.water]:
        if not os.path.exists(file_path):
            logger.error(f"Input file not found: {file_path}")
            return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        if args.country_mode:
            success = process_country_data(args.dem, args.districts, args.water, output_dir, args.resolution)
        else:
            success = process_district_data(args.district_name, args.dem, args.districts, args.water, output_dir)
        
        if success:
            logger.info("\n🎉 Data preparation completed successfully!")
            logger.info(f"📁 Output files are ready in: {output_dir.absolute()}")
            
            # List created files
            for file in sorted(output_dir.glob("*")):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    logger.info(f"   📄 {file.name} ({size_mb:.1f} MB)")
            
            logger.info("\n🚀 Next steps:")
            logger.info("   1. cd ../backend")
            logger.info("   2. ./start.sh")
            logger.info("   3. Open http://127.0.0.1:8000 in your browser")
            
        else:
            logger.error("❌ Data preparation failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())