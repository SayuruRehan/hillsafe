#!/bin/bash
# Test script to demonstrate data preparation workflow for Sri Lanka

echo "🇱🇰 HillSafe Sri Lanka Data Preparation Test"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "prepare_data.py" ]; then
    echo "❌ Please run this script from the data_processing directory"
    exit 1
fi

echo "📋 This script demonstrates how to process real Sri Lankan datasets:"
echo "   - Copernicus DEM GLO-30 (cop_dem_srilanka.tif)"
echo "   - GADM Admin Boundaries (gadm41_LKA.gpkg)"  
echo "   - HydroRIVERS Asia clipped to Sri Lanka (water_srilanka.shp)"
echo

# Check if user has real data files
echo "🔍 Checking for real data files..."
echo

if [ -f "cop_dem_srilanka.tif" ] && [ -f "gadm41_LKA.gpkg" ] && [ -f "water_srilanka.shp" ]; then
    echo "✅ Real data files found!"
    echo
    echo "🚀 Processing entire Sri Lanka..."
    python prepare_data.py \
        --country-mode \
        --dem cop_dem_srilanka.tif \
        --districts gadm41_LKA.gpkg \
        --water water_srilanka.shp \
        --resolution 0.0008333 \
        --output-dir ../backend/data
    
    if [ $? -eq 0 ]; then
        echo
        echo "✅ Sri Lanka countrywide data ready!"
        echo "🎯 You can now start the backend and explore the entire country"
    else
        echo "❌ Data processing failed"
        exit 1
    fi

elif [ -f "cop_dem_srilanka.tif" ] && [ -f "gadm41_LKA.gpkg" ]; then
    echo "⚠️  Found DEM and districts, but missing water data"
    echo "   Proceeding with DEM and districts only..."
    echo
    echo "🚀 Processing Kandy district as example..."
    python prepare_data.py \
        --district-name "Kandy" \
        --dem cop_dem_srilanka.tif \
        --districts gadm41_LKA.gpkg \
        --water water_srilanka.shp \
        --output-dir ../backend/data
    
    if [ $? -eq 0 ]; then
        echo
        echo "✅ Kandy district data ready!"
    else
        echo "❌ Data processing failed"
        exit 1
    fi

else
    echo "ℹ️  Real data files not found. Expected files:"
    echo "   📁 cop_dem_srilanka.tif    (Copernicus DEM GLO-30)"
    echo "   📁 gadm41_LKA.gpkg        (GADM Sri Lanka boundaries)"  
    echo "   📁 water_srilanka.shp     (HydroRIVERS clipped to SL)"
    echo
    echo "💡 The system will automatically create demo data when you start the backend."
    echo
    echo "🔗 Data Sources:"
    echo "   • Copernicus DEM: https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model"
    echo "   • GADM: https://gadm.org/download_country.html"
    echo "   • HydroRIVERS: https://www.hydrosheds.org/products/hydrorivers"
    echo
    
    echo "🚀 Starting backend with demo data..."
fi

echo
echo "🎉 Next steps:"
echo "   1. cd ../backend"  
echo "   2. ./start.sh"
echo "   3. Open http://127.0.0.1:8000 in your browser"
echo "   4. Click anywhere on the Sri Lanka map!"