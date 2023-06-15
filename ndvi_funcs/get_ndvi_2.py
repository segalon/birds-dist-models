
import pandas as pd
import ee
import datetime

from src.models import *
from src.utils import *
from src.config import *

import os

path_json = "/Users/user/.config/gcloud/application_default_credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_json


def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['longitude', 'latitude','datetime',  *list_of_bands]]

    return df


#project_name = ""
#ee.Authenticate()
ee.Initialize(project=project_name)


# Load the Sentinel-2 image collection
imageCollection = ee.ImageCollection("COPERNICUS/S2")

# Filter the image collection by date, cloud cover, and MGRS tile
image = imageCollection.filterDate('2018-01-01','2019-12-31')
# image = imageCollection.filterDate('2018-01-01','2020-01-03 \
#                         .sort("CLOUD_COVER", False) \
#                         .filterMetadata('MGRS_TILE','equals','36RXU')

# Load the polygon (Choose the Negev Polygon)
polygon = ee.Geometry.Polygon(
        [[[34.744223617837086, 30.39290717159926],
          [34.887045883462086, 30.407120956764057],
          [34.887045883462086, 30.44974989567197],
          [34.766196274087086, 30.454485294296127]]])

# Filter the Sentinel-2 image collection by polygon and date, and select bands B8 and B4
S2 = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filterBounds(polygon) \
        .filterDate('2018-01-01','2020-01-03') \
        .select('B8','B4')

   
S2 = S2.map(lambda image: image.addBands(image.normalizedDifference(['B8','B4']) \
                                            .select([0],['NDVI'])))
S2_NDVI= S2.select('NDVI')

#NDVI_INFO=S2_NDVI.getRegion(polygon,1000).getInfo()
NDVI_INFO=S2_NDVI.getRegion(polygon, 400).getInfo()

NDVI_DATA=ee_array_to_df(NDVI_INFO,["NDVI"])


#NDVI_DATA.groupby(['datetime']).mean('NDVI').plot()
# plt.scatter(x=NDVI_DATA.longitude,y=NDVI_DATA.latitude,c=NDVI_DATA.NDVI)
# plt.show()


NDVI_DATA['month'] = NDVI_DATA['datetime'].dt.month
NDVI_DATA['year'] = NDVI_DATA['datetime'].dt.year
NDVI_DATA['day'] = NDVI_DATA['datetime'].dt.day

# save data to file
# filter to get only the earliest day in januray per year
# save data
NDVI_DATA.to_csv('data/NDVI_DATA.csv', index=False)

df_birds, df_cls, df_ar = load_data()
