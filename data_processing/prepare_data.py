#!/usr/bin/env python3
"""
Data preparation script for HillSafe.
This script shows how to prepare real GeoTIFF data for the HillSafe backend.

Example usage:
    python prepare_data.py --dem your_dem.tif --water_bodies rivers.geojson --district kandy_boundary.geojson
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clip_dem_to_district(dem_path: str, district_path: str, output_path: str):
    """Clip DEM raster to district boundary."""
    logger.info(f"Clipping DEM {dem_path} to district {district_path}")
    
    # Load district boundary
    district_gdf = gpd.read_file(district_path)
    
    # Open DEM and clip
    with rasterio.open(dem_path) as src:
        # Ensure same CRS
        if district_gdf.crs != src.crs:
            district_gdf = district_gdf.to_crs(src.crs)
        
        # Clip DEM to district
        out_image, out_transform = mask(src, district_gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # Write clipped DEM
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
    
    logger.info(f"Clipped DEM saved to {output_path}")


def compute_slope_from_dem(dem_path: str, output_path: str):
    """Compute slope in degrees from DEM."""
    logger.info(f"Computing slope from {dem_path}")
    
    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype(np.float32)
        transform = src.transform
        profile = src.profile.copy()
        
        # Get pixel resolution (assuming square pixels in meters)
        pixel_size = abs(transform[0])  # degrees
        # Convert to approximate meters (rough conversion)
        pixel_size_m = pixel_size * 111320  # meters per degree longitude
        
        # Compute gradients
        dy, dx = np.gradient(elevation)
        dy = dy / pixel_size_m
        dx = dx / pixel_size_m
        
        # Calculate slope in degrees
        slope = np.arctan(np.sqrt(dx*dx + dy*dy)) * 180.0 / np.pi
        
        # Update profile for slope output
        profile.update(dtype=slope.dtype)
        
        # Write slope
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(slope, 1)
    
    logger.info(f"Slope raster saved to {output_path}")
    logger.info(f"Slope range: {slope.min():.1f}° to {slope.max():.1f}°")


def prepare_water_bodies(water_path: str, output_path: str, district_path: str = None):
    """Prepare water bodies vector file."""
    logger.info(f"Processing water bodies from {water_path}")
    
    water_gdf = gpd.read_file(water_path)
    
    if district_path:
        # Clip to district boundary
        district_gdf = gpd.read_file(district_path)
        if water_gdf.crs != district_gdf.crs:
            water_gdf = water_gdf.to_crs(district_gdf.crs)
        
        water_gdf = gpd.clip(water_gdf, district_gdf)
    
    # Save processed water bodies
    water_gdf.to_file(output_path, driver="GeoJSON")
    logger.info(f"Water bodies saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare spatial data for HillSafe")
    parser.add_argument("--dem", help="Path to DEM GeoTIFF file")
    parser.add_argument("--water-bodies", help="Path to water bodies vector file (GeoJSON/Shapefile)")
    parser.add_argument("--district", help="Path to district boundary vector file")
    parser.add_argument("--output-dir", default="./data", help="Output directory for processed files")
    parser.add_argument("--compute-slope", action="store_true", help="Compute slope from DEM")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.dem:
        dem_output = output_dir / "dem.tif"
        
        if args.district:
            # Clip DEM to district
            clip_dem_to_district(args.dem, args.district, str(dem_output))
        else:
            # Just copy DEM
            import shutil
            shutil.copy2(args.dem, dem_output)
            logger.info(f"Copied DEM to {dem_output}")
        
        if args.compute_slope:
            # Compute slope from (clipped) DEM
            slope_output = output_dir / "slope.tif"
            compute_slope_from_dem(str(dem_output), str(slope_output))
    
    if args.water_bodies:
        water_output = output_dir / "water_bodies.geojson"
        prepare_water_bodies(args.water_bodies, str(water_output), args.district)
    
    if args.district:
        # Copy district boundary for reference
        district_output = output_dir / "district_boundary.geojson"
        district_gdf = gpd.read_file(args.district)
        district_gdf.to_file(str(district_output), driver="GeoJSON")
        logger.info(f"District boundary saved to {district_output}")
    
    logger.info("Data preparation complete!")
    logger.info(f"Files ready in: {output_dir}")
    
    # List created files
    for file in output_dir.glob("*"):
        if file.is_file():
            logger.info(f"  - {file.name}")


if __name__ == "__main__":
    main()