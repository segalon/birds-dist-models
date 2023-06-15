
import pandas as pd
import ee
import datetime

from src.models import *
from src.utils import *
from src.config import *

import os

path_json = "/Users/user/.config/gcloud/application_default_credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_json

df_birds, df_cls, df_ar = load_data()

# Authenticate and initialize the Earth Engine API
#project_name = ""
#ee.Authenticate()
ee.Initialize(project=project_name)

# Convert your DataFrame's geometry column to a list of shapely geometries
shapely_geometries = df_ar['geometry'].tolist()

# Convert the shapely geometries to Earth Engine geometries
ee_geometries = [ee.Geometry.Polygon(list(polygon.exterior.coords)) for polygon in shapely_geometries]

# Define the date for NDVI data
date = '2019-01-01'
date_end = '2019-01-31'

# Load the MODIS NDVI dataset
modis_ndvi = ee.ImageCollection('MODIS/006/MOD13A1').select('NDVI')

# Filter the data to the specified date range
ndvi_filtered = modis_ndvi.filterDate(date, date_end)

# Calculate the mean NDVI value for the specified date range
def get_ndvi_values(ee_geometries, ndvi_filtered):
    ndvi_values = []

    for idx, geometry in enumerate(ee_geometries):
        print(f"Processing polygon {idx + 1} of {len(ee_geometries)}")
        ndvi_mean = ndvi_filtered.mean().reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=1000)
        ndvi_mean = ndvi_mean.getInfo()['NDVI']
        ndvi_values.append(ndvi_mean)

    return ndvi_values

ndvi_values = get_ndvi_values(ee_geometries, ndvi_filtered)

# Add the NDVI values to the DataFrame
df_ar['ndvi_january_2019'] = ndvi_values
